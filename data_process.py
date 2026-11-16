import jsonlines
import os
import torch
from typing import Any
from code_minification import minify_cpp_function
from transformers import AutoTokenizer


def get_code_boundary_token_ids(tokenizer) -> set[int]:
    """
    Precomputes vocabulary token IDs containing structural C/C++ boundaries
    ('\\n', ';', '}') so boundary scanning works in O(1) on slow and fast tokenizers.
    """
    boundary_ids = set()
    for tid in range(len(tokenizer)):
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        if "\n" in decoded or ";" in decoded or "}" in decoded:
            boundary_ids.add(tid)
    return boundary_ids


def chunk_code_with_boundaries(
    text: str,
    tokenizer,
    boundary_token_ids: set[int],
    max_chunk_size: int,
    stride: int,
) -> list[list[int]]:
    """
    Tokenizes C/C++ code and splits it into overlapping chunks, snapping cuts backward
    to the nearest newline, semicolon, or closing brace without requiring `offset_mapping`.
    """
    num_special_tokens = len(tokenizer.build_inputs_with_special_tokens([]))
    effective_max = max_chunk_size - num_special_tokens
    overlap = max(0, effective_max - stride)

    input_ids = tokenizer.encode(text, add_special_tokens=False)

    # Guarantee at least one valid [CLS][SEP] chunk even if the input string is empty
    if not input_ids:
        return [tokenizer.build_inputs_with_special_tokens([])]

    chunks = []
    start_idx = 0
    total_tokens = len(input_ids)

    while start_idx < total_tokens:
        end_idx = start_idx + effective_max
        if end_idx >= total_tokens:
            chunks.append(input_ids[start_idx:total_tokens])
            break

        # Scan backward up to 50% of the chunk window for a clean structural boundary
        best_cut_idx = end_idx
        min_cut_idx = start_idx + (effective_max // 2)
        for i in range(end_idx - 1, min_cut_idx, -1):
            if input_ids[i] in boundary_token_ids:
                best_cut_idx = i + 1
                break

        chunks.append(input_ids[start_idx:best_cut_idx])

        next_start = best_cut_idx - overlap
        start_idx = max(start_idx + 1, next_start)

    return [tokenizer.build_inputs_with_special_tokens(c) for c in chunks]


# --- Dataset Preparation (Minification + Boundary Chunking + Head-and-Tail Truncation) ---
def prepare_mil_dataset(
        example: dict[str, Any], 
        tokenizer: AutoTokenizer, 
        seq_len: int, 
        stride: int,
        max_chunks_per_bag: int,
        boundary_ids: set[int]
    ) -> dict[str, list[Any]]:

    # Preserve newlines (`aggressive_whitespace=False`) so structural boundaries survive
    minified = minify_cpp_function(
        example["func"], 
        aggressive_whitespace=False,
        rename_long_identifiers=False
    )

    chunked_ids = chunk_code_with_boundaries(
        text=minified,
        tokenizer=tokenizer,
        boundary_token_ids=boundary_ids,
        max_chunk_size=seq_len,
        stride=stride,
    )

    # Head-and-Tail truncation if a function still exceeds MAX_CHUNKS_PER_BAG
    if len(chunked_ids) > max_chunks_per_bag:
        print(f"Function {example['idx']} truncated from {len(chunked_ids)} to {max_chunks_per_bag} chunks (Head-and-Tail).")
        half = max_chunks_per_bag // 2
        chunked_ids = chunked_ids[:half] + chunked_ids[-half:]

    # attention_masks = [[1] * len(chunk) for chunk in chunked_ids]

    return {
        "input_ids": chunked_ids,
        # "attention_mask": attention_masks,
        "label": example["target"]
    }


def select_columns(src_file: str, dst_file: str, columns: list[str]) -> None:
    """
    Selects specific columns from a JSONL file and writes them to a new JSONL file.

    Args:
        src_file (str): Path to the source JSONL file.
        dst_file (str): Path to the destination JSONL file.
        columns (list): List of column names to select.
    """
    with jsonlines.open(src_file) as reader:
        with jsonlines.open(dst_file, mode='w') as writer:
            for obj in reader:
                selected_obj = {col: obj[col] for col in columns}
                writer.write(selected_obj)


def main():
    file_names = [
        "primevul_train.jsonl",
        "primevul_valid.jsonl",
        "primevul_test.jsonl",
        "primevul_train_paired.jsonl",
        "primevul_valid_paired.jsonl",
        "primevul_test_paired.jsonl"
    ]
    src_dir = "data/origin/primevul"
    dst_dir = "data/processed/primevul"
    selected_columns = ["idx", "func", "commit_id", "hash", "target"]

    os.makedirs(dst_dir, exist_ok=True)

    for file_name in file_names:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        select_columns(src_file, dst_file, columns=selected_columns)
        print(f"Processed {file_name} and saved to {dst_file}")


if __name__ == "__main__":
    main()