from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from collator import AttentionMILCollator
from data_process import prepare_mil_dataset, get_code_boundary_token_ids
import logging
import difflib
import torch
import torch.nn.functional as F


logger = logging.getLogger("codesage_ft.playground")
logger.setLevel(logging.INFO)
for handler in (logging.StreamHandler(), logging.FileHandler("playground.log")):
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.info("---- Playground Script for CodeSage Fine-Tuning ----")


encoder_name = "codesage/codesage-large-v2"
data_files = {
    "train": r"data/processed/primevul/primevul_train_paired.jsonl",
    "validation": r"data/processed/primevul/primevul_valid_paired.jsonl",
    "test": r"data/processed/primevul/primevul_test_paired.jsonl"
}


def dummy_test(model):
    logger.info("Testing the model with a dummy input...")
    logger.info(model)
    dummy_input = torch.randint(0, 100, (1, 16), dtype=torch.int32)
    dummy_attention = torch.ones_like(dummy_input, dtype=torch.int32)

    outputs = model(input_ids=dummy_input, attention_mask=dummy_attention)
    logger.info(outputs.pooler_output.shape)  # Should be (1, hidden_size)


def load_model(dbg=False):
    logger.info("Loading the model...")

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
    )
    return base_encoder


def load_tokenizer():
    logger.info("Loading the tokenizer...")
    return AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)


def measure_twin_similarity(encoder, dataset):
    for i in range(0, len(dataset), 2):
        assert dataset[i]["commit_id"] == dataset[i+1]["commit_id"], f"Commit IDs do not match for samples {i} and {i+1}"

    dataset = dataset.map(
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collator)

    encoder.eval()  # Set the model to evaluation mode

    cos_similarities_per_pair = torch.zeros(len(dataloader), dtype=torch.float32)
    l2_distances_per_pair = torch.zeros(len(dataloader), dtype=torch.float32)

    for idx, batch in enumerate(dataloader):
        batch = {k: v.to(encoder.device) for k, v in batch.items()}  # Move batch to the same device as the model
        
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            B, C, L = batch["input_ids"].shape
            logger.info(f"Batch {idx}: Input shape: {batch['input_ids'].shape}, Attention mask shape: {batch['attention_mask'].shape}, Chunk mask shape: {batch['chunk_mask'].shape}")
            out = encoder(
                input_ids=batch["input_ids"].view(-1, L),
                attention_mask=batch["attention_mask"].view(-1, L),
            )
            tok_emb = out.last_hidden_state.float()
            mask = batch["attention_mask"].view(-1, L, 1).float()
            pooled = ((tok_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).view(B, C, -1)

            if B != 2:
                logger.info(f"Warning: Expected batch size of 2 for paired samples, but got {B}. Skipping similarity computation for this batch.")
                continue

            # Compare chunk 0 of vuln (index 0) vs chunk 0 of patch (index 1)
            cos_sim = F.cosine_similarity(pooled[0, 0], pooled[1, 0], dim=-1).item()
            l2_dist = torch.norm(pooled[0, 0] - pooled[1, 0]).item()
            rel_diff = l2_dist / torch.norm(pooled[0, 0]).item()

        cos_similarities_per_pair[idx] = cos_sim
        l2_distances_per_pair[idx] = l2_dist

        logger.info(f"Batch {idx} - Cosine Similarity : {cos_sim:.6f}, L2 Distance : {l2_dist:.4f} (Relative diff: {rel_diff:.2%})")

    logger.info(f"Cosine Similarity across all pairs:\n\tmean: {cos_similarities_per_pair.mean().item():.4f}, std: {cos_similarities_per_pair.std().item():.4f}, min: {cos_similarities_per_pair.min().item():.4f}, max: {cos_similarities_per_pair.max().item():.4f}")
    logger.info(f"L2 Distance across all pairs:\n\tmean: {l2_distances_per_pair.mean().item():.4f}, std: {l2_distances_per_pair.std().item():.4f}, min: {l2_distances_per_pair.min().item():.4f}, max: {l2_distances_per_pair.max().item():.4f}")


def probe_all_layers_twin_separation(
    mil_model,
    dataloader,
    num_pairs: int = 16,
):
    """
    Sweeps all 25 hidden states (`emb` + `blocks 0..23`) across `num_pairs`
    to measure where `codesage-large-v2` separates `vuln` and `patch` twins
    both BEFORE pooling (aligned shared tokens) and AFTER mean pooling.
    """
    mil_model.eval()
    device = next(mil_model.parameters()).device

    num_layers = 25  # 0 = embeddings, 1..24 = CodeSageBlock 0..23
    pre_pool_rel_l2 = [[] for _ in range(num_layers)]
    post_pool_rel_l2 = [[] for _ in range(num_layers)]
    post_pool_cos = [[] for _ in range(num_layers)]

    collected = 0
    for batch in dataloader:
        if collected >= num_pairs:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        B, C, L = batch["input_ids"].shape

        # Single forward pass returning all 25 hidden states (no grad -> ~3.4 GiB VRAM)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = mil_model.encoder(
                input_ids=batch["input_ids"].view(-1, L),
                attention_mask=batch["attention_mask"].view(-1, L),
                output_hidden_states=True,
            )
            # Stack to (25, B, C, L, H) in float32
            all_h = torch.stack([h.float() for h in out.hidden_states], dim=0).view(
                num_layers, B, C, L, -1
            )

        mask_f = batch["attention_mask"].float()  # (B, C, L)
        tok_counts = mask_f.sum(dim=-1).clamp(min=1.0).unsqueeze(-1)  # (B, C, 1)

        # Process each `(vuln, patch)` pair in the batch (`2*p` and `2*p + 1`)
        for p in range(B // 2):
            if collected >= num_pairs:
                break
            v_idx, p_idx = 2 * p, 2 * p + 1

            # Select a modified `vuln` chunk (`diff_mask == 1.0`)
            if "diff_mask" in batch and (batch["diff_mask"][v_idx] > 0).any():
                c_v = int(torch.argmax(batch["diff_mask"][v_idx]).item())
            else:
                continue  # Skip truncated pairs with no visible diff

            len_v = int(batch["attention_mask"][v_idx, c_v].sum().item())
            ids_v = batch["input_ids"][v_idx, c_v, :len_v].tolist()

            # Match to the `patch` chunk with highest token overlap
            patch_valid = int(batch["chunk_mask"][p_idx].sum().item())
            best_c_p, best_ratio = 0, -1.0
            for c in range(patch_valid):
                l_cand = int(batch["attention_mask"][p_idx, c].sum().item())
                ids_cand = batch["input_ids"][p_idx, c, :l_cand].tolist()
                ratio = difflib.SequenceMatcher(None, ids_v, ids_cand, autojunk=False).quick_ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_c_p = c

            len_p = int(batch["attention_mask"][p_idx, best_c_p].sum().item())
            ids_p = batch["input_ids"][p_idx, best_c_p, :len_p].tolist()

            # Align identical tokens between the two chunks
            matcher = difflib.SequenceMatcher(None, ids_v, ids_p, autojunk=False)
            aligned_v, aligned_p = [], []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    aligned_v.extend(range(i1, i2))
                    aligned_p.extend(range(j1, j2))

            if not aligned_v:
                continue

            idx_v_t = torch.tensor(aligned_v, device=device)
            idx_p_t = torch.tensor(aligned_p, device=device)

            # Evaluate all 25 layers simultaneously
            for layer_idx in range(num_layers):
                # 1. Pre-pooling: mean relative L2 shift across aligned shared tokens
                hv_tok = all_h[layer_idx, v_idx, c_v, idx_v_t]      # (N_aligned, H)
                hp_tok = all_h[layer_idx, p_idx, best_c_p, idx_p_t] # (N_aligned, H)
                tok_rel = (
                    torch.norm(hv_tok - hp_tok, dim=-1)
                    / torch.norm(hv_tok, dim=-1).clamp(min=1e-8)
                ).mean().item()
                pre_pool_rel_l2[layer_idx].append(tok_rel)

                # 2. Post-mean-pooling: chunk-level separation
                hv_pool = (
                    all_h[layer_idx, v_idx, c_v] * mask_f[v_idx, c_v].unsqueeze(-1)
                ).sum(dim=0) / tok_counts[v_idx, c_v]
                hp_pool = (
                    all_h[layer_idx, p_idx, best_c_p] * mask_f[p_idx, best_c_p].unsqueeze(-1)
                ).sum(dim=0) / tok_counts[p_idx, best_c_p]

                pool_rel = (
                    torch.norm(hv_pool - hp_pool) / torch.norm(hv_pool).clamp(min=1e-8)
                ).item()
                pool_cos = F.cosine_similarity(hv_pool, hp_pool, dim=0).item()

                post_pool_rel_l2[layer_idx].append(pool_rel)
                post_pool_cos[layer_idx].append(pool_cos)

            collected += 1

    print(f"Averaged across {collected} modified `(vuln, patch)` pairs:\n")
    print(f"{'Layer':<14} | {'Pre-Pool Aligned Rel L2':<24} | {'Post-Pool Rel L2':<18} | {'Post-Pool Cos'}")
    print("-" * 78)
    for l_idx in range(num_layers):
        name = "0 (emb)" if l_idx == 0 else f"{l_idx} (block {l_idx - 1})"
        pre_m = sum(pre_pool_rel_l2[l_idx]) / max(len(pre_pool_rel_l2[l_idx]), 1)
        post_m = sum(post_pool_rel_l2[l_idx]) / max(len(post_pool_rel_l2[l_idx]), 1)
        cos_m = sum(post_pool_cos[l_idx]) / max(len(post_pool_cos[l_idx]), 1)
        print(f"{name:<14} | {pre_m:>22.2%} | {post_m:>16.2%} | {cos_m:>13.6f}")


tokenizer = load_tokenizer()
encoder = load_model()
logger.info(encoder)

dataset: DatasetDict = load_dataset("json", data_files=data_files) # type: ignore
measure_twin_similarity(encoder, dataset["train"])
