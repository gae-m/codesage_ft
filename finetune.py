import argparse
import os
import sys
from typing import cast
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.metrics import roc_auc_score
from time import time, perf_counter

from collator import AttentionMILCollator
from classifier_wrapper import AttentionMILCodeSage
# from cs_checkpointing import enable_codesage_block_checkpointing
from code_minification import minify_cpp_function
from data_process import get_code_boundary_token_ids, prepare_mil_dataset


ENCODER = "codesage/codesage-large-v2"
CHECKPOINTING_RATIO = 1.0           # Checkpoint only this fraction of CodeSage blocks to balance memory and speed
MAX_CHUNKS_PER_BAG = 20             # Limit the number of chunks per function to avoid memory issues
MAX_SEQ_LEN = 2048                  # Each chunk is a sequence of 512 tokens. If you have enough GPU memory, you can increase this to 1024 or 2048, but be aware of the memory trade-offs.
OVERLAP = MAX_SEQ_LEN // 4          # Overlap between chunks to ensure context is preserved across chunk boundaries. This is set to 25% of the chunk size, but you can adjust it based on your needs.
TRAINING_BATCH_SIZE = 1             # Each batch contains 1 function, which may consist of multiple chunks. Adjust based on your GPU memory.
STRIDE = MAX_SEQ_LEN - OVERLAP
DEBUG = False                       # Set to True for quick testing with a smaller dataset and fewer epochs. This is useful for debugging and ensuring your code runs without errors before committing to a full training run.
EPOCHS = 5                          # Number of training epochs. You can increase this for better performance, but it will take longer to train.
DATALOADER_WORKERS = 8              # Number of subprocesses to use for data loading. Adjust based on your CPU cores and memory.

# Define the mapping of your splits to the local file paths
DATAPATHS = {
    "train": r"data/processed/primevul/primevul_train_paired.jsonl",
    "validation": r"data/processed/primevul/primevul_valid_paired.jsonl",
    "test": r"data/processed/primevul/primevul_test_paired.jsonl"
}

# def load_data(data_files: dict[str, str], tokenizer: AutoTokenizer) -> DatasetDict:
#     # Load the dataset using the 'json' builder
#     raw_dataset: DatasetDict = load_dataset("json", data_files=data_files)
#     print(raw_dataset)

#     def minify_dataset(example: dict[str, Any]) -> dict[str, Any]:
#         # Apply code minification to the 'func' field
#         minified_func = minify_cpp_function(
#             example["func"], 
#             aggressive_whitespace=False,
#             rename_long_identifiers=False
#         )
#         example["func"] = minified_func
#         return example

#     min_dataset = raw_dataset.map(minify_dataset)

#     # TODO: Snap to Structural Boundaries (Crucial)
#     # Standard Hugging Face chunking slices at exact token counts. Slicing source code at exactly token 1024 might cut a variable name in half (e.g., buffer becomes buf and fer) or split an arithmetic condition (if (x <  |  y)).
#     # To fix this, write a custom chunking wrapper that calculates the theoretical stride (e.g., 768), but then scans backward in the original text to snap the cut to the nearest newline (\n) or statement terminator (;, }).

#     def tokenize_into_bags(example: dict[str, Any]) -> dict[str, Any]:
#         # 1. Tokenize the entire function without truncation and without special tokens
#         tokenized = tokenizer(
#             example["func"], 
#             truncation=False, 
#             padding=False,
#             add_special_tokens=False 
#         )
        
#         raw_input_ids = tokenized["input_ids"]
        
#         # 2. Identify special tokens for CodeSage (defaults to <s>/</s> or [CLS]/[SEP])
#         cls_token = tokenizer.cls_token_id or tokenizer.bos_token_id
#         sep_token = tokenizer.sep_token_id or tokenizer.eos_token_id
#         pad_token = tokenizer.pad_token_id
        
#         max_length = MAX_SEQ_LEN
#         overlap = OVERLAP
#         usable_length = max_length - 2  # Reserve 2 spaces for the CLS and SEP tokens
#         step = usable_length - overlap
        
#         chunked_input_ids = []
#         chunked_attention_mask = []
        
#         # 3. Slide the window to create properly formatted 2D chunks
#         for i in range(0, len(raw_input_ids), step):
#             chunk_code_ids = raw_input_ids[i : i + usable_length]
            
#             # Manually inject [CLS] at the start and [SEP] at the end
#             chunk_ids = [cls_token] + chunk_code_ids + [sep_token]
#             chunk_mask = [1] * len(chunk_ids)
            
#             # Pad the individual chunk to exactly MAX_SEQ_LEN tokens
#             # TODO: review this padding logic to ensure it doesn't introduce unnecessary padding if the chunk is already at max length
#             pad_len = max_length - len(chunk_ids)
#             if pad_len > 0:
#                 chunk_ids.extend([pad_token] * pad_len)
#                 chunk_mask.extend([0] * pad_len)
                
#             chunked_input_ids.append(chunk_ids)
#             chunked_attention_mask.append(chunk_mask)
                
#         # Handle the edge case of an empty string
#         if not chunked_input_ids:
#             chunk_ids = [cls_token, sep_token] + [pad_token] * (max_length - 2)
#             chunk_mask = [1, 1] + [0] * (max_length - 2)
#             chunked_input_ids.append(chunk_ids)
#             chunked_attention_mask.append(chunk_mask)

#         if len(chunked_input_ids) > MAX_CHUNKS_PER_BAG:
#             print(f"Warning: Function {example['idx']} has {len(chunked_input_ids)} chunks, exceeding the limit of {MAX_CHUNKS_PER_BAG}. Head-and-Tail truncation to {MAX_CHUNKS_PER_BAG} chunks.")
#             chunked_input_ids = chunked_input_ids[:MAX_CHUNKS_PER_BAG//2] + chunked_input_ids[-(MAX_CHUNKS_PER_BAG//2):]
#             chunked_attention_mask = chunked_attention_mask[:MAX_CHUNKS_PER_BAG//2] + chunked_attention_mask[-(MAX_CHUNKS_PER_BAG//2):]
#             # chunked_input_ids = chunked_input_ids[:MAX_CHUNKS_PER_BAG]
#             # chunked_attention_mask = chunked_attention_mask[:MAX_CHUNKS_PER_BAG]
            
#         return {
#             "input_ids": chunked_input_ids,
#             "attention_mask": chunked_attention_mask,
#             "label": example["target"]
#         }

#     # Apply without batched=True to keep chunks grouped by their parent function
#     tokenized_dataset = min_dataset.map(tokenize_into_bags, remove_columns=["func", "commit_id", "target", "hash", "idx"])
#     return tokenized_dataset


def compute_auroc_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Trainer collects every model output other than loss as predictions.
    # The wrapper also returns attention_weights, so predictions can be
    # (logits, attention_weights) rather than just the logits tensor.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    # 1. Apply sigmoid to convert raw logits into probabilities
    probabilities = (
        torch.sigmoid(torch.as_tensor(logits))
        .squeeze(-1)
        .reshape(-1)
        .numpy()
    )
    labels = torch.as_tensor(labels).reshape(-1).numpy()
    
    # 2. Calculate AUROC
    try:
        auroc = roc_auc_score(labels, probabilities)
    except ValueError:
        # Handles edge cases where a batch might only contain one class during testing
        auroc = 0.5 
        
    return {"auroc": auroc}


def save_model(model: AttentionMILCodeSage, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save only the LoRA adapter weights (very lightweight, ~10-50MB)
    encoder = cast(PeftModel, model.encoder)  # Ensure we treat the encoder as a PeftModel
    encoder.save_pretrained(os.path.join(out_dir, "lora-adapters"))

    # 2. Save the state dictionaries of your custom Attention and Classifier networks
    custom_weights = {
        "attention_state": model.attention.state_dict(),
        "classifier_state": model.classifier.state_dict()
    }
    torch.save(custom_weights, os.path.join(out_dir, "custom_heads.pt"))


def fine_tuning(data_files: dict[str, str], out_dir: str, encoder_name: str = "codesage/codesage-large-v2", dbg = False):
    print("Debug mode: ", dbg)
    print("Encoder: ", encoder_name)
    print(f"{MAX_CHUNKS_PER_BAG=}\n{MAX_SEQ_LEN=}\n{OVERLAP=}\n{TRAINING_BATCH_SIZE=}\n{STRIDE=}\n{CHECKPOINTING_RATIO=}")
    
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

    print("Loading and tokenizing dataset...")
    raw_dataset: DatasetDict = load_dataset("json", data_files=data_files) # type: ignore
    tokenized_dataset = raw_dataset.map(
        prepare_mil_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "seq_len": MAX_SEQ_LEN,
            "stride": STRIDE,
            "max_chunks_per_bag": MAX_CHUNKS_PER_BAG,
            "boundary_ids": get_code_boundary_token_ids(tokenizer)
        },
        remove_columns=["func", "commit_id", "target", "hash", "idx"],
        keep_in_memory=True,
        num_proc=DATALOADER_WORKERS
    )
    print("Tokenization complete. Dataset overview:")
    print(tokenized_dataset)

    if dbg:
        print("Debug mode: Reducing dataset size for quick testing...")
        train_set = tokenized_dataset["train"].select(range(min(10, len(tokenized_dataset["train"]))))
        eval_set = tokenized_dataset["validation"].select(range(min(10, len(tokenized_dataset["validation"]))))
        test_set = tokenized_dataset["test"].select(range(min(10, len(tokenized_dataset["test"]))))
        tokenized_dataset = DatasetDict({"train": train_set, "validation": eval_set, "test": test_set})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_support = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print("Using device:", device)
    print("bf16 support:", bf16_support)

    print("Loading the model...")
    config = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
    config.attention_dropout_prob = 0.0
    config.residual_dropout_prob = 0.0
    config.embedding_dropout_prob = 0.0

    # Load base CodeSage without the classification head
    base_encoder = AutoModel.from_pretrained(
        encoder_name,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16 if bf16_support else torch.float32
        # Loads the frozen weights directly in 16-bit
    )

    # Apply LoRA to the base encoder
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",  # Apply LoRA to all linear layers in the encoder
        fan_in_fan_out=True,
        bias="none"
        # Do NOT include "classifier" here, because our classifier is in the wrapper, not the PEFT model
    )

    peft_encoder = get_peft_model(base_encoder, lora_config)

    # Initialize the Attention-MIL model
    mil_model = AttentionMILCodeSage(encoder=peft_encoder, hidden_size=base_encoder.config.hidden_size)

    print("Model loaded. Preparing for training...")

    mil_model = mil_model.to(device) # Move to GPU

    # Create the Trainer
    training_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=3 if dbg else -1,
        group_by_length=True,                             # Useful only if batch_size > 1, like for the eval phase in this case
        per_device_train_batch_size=TRAINING_BATCH_SIZE,  # Keep batch size small; you are processing (batch * chunks) sequences at once
        per_device_eval_batch_size=4,                     # You can increase this if you have enough GPU memory
        learning_rate=2e-4,
        gradient_accumulation_steps=8,                    # Simulate a larger batch size
        gradient_checkpointing=True,                      # Save memory by not storing intermediate activations. CodeSage does not support gradient checkpointing natively.
        gradient_checkpointing_kwargs={"checkpoint_ratio": CHECKPOINTING_RATIO}, # Checkpoint only CHECKPOINTING_RATIO of the CodeSage blocks to balance memory and speed
        warmup_ratio=0.1,
        num_train_epochs=EPOCHS,
        bf16=bf16_support,
        # Logging and evaluation strategies
        logging_steps=50,
        logging_first_step=True,
        report_to=["tensorboard"],
        logging_dir=out_dir,
        # Save strategies
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        # Optional: still log validation metrics per epoch without altering model selection
        eval_strategy="epoch",
        remove_unused_columns=False,
        # Dataloaders ...
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True
    )

    # While CodeSage does have a pad token, some code tokenizers (like GPT-2 base) do not have one set by default.
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset["train"] = concatenate_datasets([tokenized_dataset["train"], tokenized_dataset["validation"]])

    trainer = Trainer(
        model=mil_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=AttentionMILCollator(pad_token_id=tokenizer.pad_token_id, empty_chunk_ids=tokenizer.build_inputs_with_special_tokens([])),
        compute_metrics=compute_auroc_metrics
    )

    print("Starting training...")
    start_time = perf_counter()
    try:
        trainer.train(
            ignore_keys_for_eval=["attention_weights"]  # Ignore the attention weights during evaluation to avoid unnecessary computation
        )
        end_time = perf_counter()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
    except KeyboardInterrupt:
        end_time = perf_counter()
        print(f"Training interrupted by user after {end_time - start_time:.2f} seconds.")

    # 2. Evaluate on the Test Set
    print("Running final evaluation on the Test Set...")
    test_results = trainer.evaluate(tokenized_dataset["test"], ignore_keys=["attention_weights"]) # type: ignore

    # 3. Print the comprehensive metrics
    for metric_name, value in test_results.items():
        print(f"{metric_name}: {value:.4f}")

    # Save only the adapters and the classifier head
    print(f"Saving the fine-tuned model to {out_dir}...")
    save_model(mil_model, out_dir)


def load_model(model_dir: str, encoder_name: str = "codesage/codesage-large-v2") -> AttentionMILCodeSage:
    # 1. Load the base CodeSage model
    base_model = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

    # 2. Load the trained LoRA adapters onto the base model
    peft_encoder = PeftModel.from_pretrained(base_model, os.path.join(model_dir, "lora-adapters"))

    # 3. Initialize your custom architecture wrapper
    loaded_mil_model = AttentionMILCodeSage(
        encoder=peft_encoder, 
        hidden_size=base_model.config.hidden_size, 
        attention_dim=256
    )

    # 4. Load the custom attention and classification heads
    custom_weights = torch.load(os.path.join(model_dir, "custom_heads.pt"))
    loaded_mil_model.attention.load_state_dict(custom_weights["attention_state"])
    loaded_mil_model.classifier.load_state_dict(custom_weights["classifier_state"])

    # Set to evaluation mode
    loaded_mil_model.eval()

    return loaded_mil_model


class Tee:
    def __init__(self, filename, mode="a"):
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        # Ensures real-time output without buffering delays
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


enc_name_map = {
    "codesage/codesage-large-v2": "cs2l",
    "codesage/codesage-base-v2": "cs2b",
    "codesage/codesage-small-v2": "cs2s"
}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeSage with Attention-MIL on a custom dataset.")
    parser.add_argument("--dbg", "-d", action="store_true", default=DEBUG, help="Enable debug mode for quick testing.")

    args = parser.parse_args()

    # Define the output directory for saving the model
    experiment_name = f"{enc_name_map[ENCODER]}_mil_ft_{int(time())}"
    if args.dbg:
        experiment_name = 'dbg_' + experiment_name

    out_dir = f"out/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    tee = Tee(os.path.join(out_dir, "train.log"))
    sys.stdout = tee

    # Fine-tune the model
    fine_tuning(DATAPATHS, out_dir, encoder_name=ENCODER, dbg=args.dbg)

    # Restore original stdout when done
    tee.close()


if __name__ == "__main__":
    main()
