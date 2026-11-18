from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from collator import AttentionMILCollator
from data_process import prepare_mil_dataset, get_code_boundary_token_ids


encoder_name = "codesage/codesage-large-v2"
data_files = {
    "train": r"data/processed/primevul/primevul_train_paired.jsonl",
    "validation": r"data/processed/primevul/primevul_valid_paired.jsonl",
    "test": r"data/processed/primevul/primevul_test_paired.jsonl"
}


def dummy_test(model):
    print("Testing the model with a dummy input...")
    print(model)
    dummy_input = torch.randint(0, 100, (1, 16), dtype=torch.int32)
    dummy_attention = torch.ones_like(dummy_input, dtype=torch.int32)

    outputs = model(input_ids=dummy_input, attention_mask=dummy_attention)
    print(outputs.pooler_output.shape)  # Should be (1, hidden_size)


def load_model(dbg=False):
    print("Loading the model...")

    config = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
    config.attention_dropout_prob = 0.0
    config.residual_dropout_prob = 0.0
    config.embedding_dropout_prob = 0.0

    if dbg:
        config.num_hidden_layers = 1
        config.num_attention_heads = 1
        return AutoModel.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # Load base CodeSage without the classification head
    base_encoder = AutoModel.from_pretrained(
        encoder_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,  # Loads the frozen weights directly in 16-bit
        device_map="auto"  # Automatically places layers on available devices (GPU/CPU)
    )
    return base_encoder


def load_tokenizer():
    print("Loading the tokenizer...")
    return AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)


tokenizer = load_tokenizer()
encoder = load_model(dbg = True)
dataset: DatasetDict = load_dataset("json", data_files=data_files) # type: ignore

play_dataset = dataset["test"].select(range(10))  # Select the first 10 samples for testing

for i in range(0, len(play_dataset), 2):
    assert play_dataset[i]["commit_id"] == play_dataset[i+1]["commit_id"], f"Commit IDs do not match for samples {i} and {i+1}"

play_dataset = play_dataset.map(
    prepare_mil_dataset,
    fn_kwargs={
        "tokenizer": tokenizer,
        "seq_len": 2048,
        "stride": 512*3,
        "max_chunks_per_bag": 30,
        "boundary_ids": get_code_boundary_token_ids(tokenizer),
    },
    keep_in_memory=True
)

collator = AttentionMILCollator(pad_token_id=tokenizer.pad_token_id, empty_chunk_ids=tokenizer.build_inputs_with_special_tokens([]))
dataloader = torch.utils.data.DataLoader(play_dataset, batch_size=2, collate_fn=collator)

# args = TrainingArguments(
#     output_dir="output",
#     per_device_train_batch_size=2,
#     bf16=True
# )

# trainer = Trainer(model=encoder, args=args, data_collator=collator)

# trainer.predict(play_dataset)  # Run predictions on the selected dataset

encoder.eval()  # Set the model to evaluation mode

# embeds = []
# chunk_masks = []

for idx, batch in enumerate(dataloader):
    batch = {k: v.to(encoder.device) for k, v in batch.items()}  # Move batch to the same device as the model
    
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        B, C, L = batch["input_ids"].shape
        print(f"Batch {idx}: Input shape: {batch['input_ids'].shape}, Attention mask shape: {batch['attention_mask'].shape}, Chunk mask shape: {batch['chunk_mask'].shape}")
        out = encoder(
            input_ids=batch["input_ids"].view(-1, L),
            attention_mask=batch["attention_mask"].view(-1, L),
        )
        tok_emb = out.last_hidden_state.float()
        mask = batch["attention_mask"].view(-1, L, 1).float()
        pooled = ((tok_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).view(B, C, -1)

        # Compare chunk 0 of vuln (index 0) vs chunk 0 of patch (index 1)
        cos_sim = F.cosine_similarity(pooled[0, 0], pooled[1, 0], dim=-1).item()
        l2_dist = torch.norm(pooled[0, 0] - pooled[1, 0]).item()
        rel_diff = l2_dist / torch.norm(pooled[0, 0]).item()

    print(f"Batch {idx}: Cosine Similarity : {cos_sim:.6f}")
    print(f"Batch {idx}: L2 Distance       : {l2_dist:.4f} (Relative diff: {rel_diff:.2%})")

# for emb, chunk_mask in zip(embeds, chunk_masks):
#     print(emb.shape)
#     x = emb[0, chunk_mask[0].bool(), :]
#     y = emb[1, chunk_mask[1].bool(), :]

#     flat_x = x.view(-1)
#     flat_y = y.view(-1)

#     print(f"Flat x shape: {flat_x.shape}, Flat y shape: {flat_y.shape}")

#     print("flat_x == flat_y:", torch.equal(flat_x, flat_y))

        
    
        






