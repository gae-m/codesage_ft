import difflib
import logging
from typing import Any
from code_minification import minify_cpp_function
from data_process import get_code_boundary_token_ids


logger = logging.getLogger(__name__)


def get_changed_char_spans(
    vuln_code: str,
    patch_code: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Computes character intervals `(start, end)` in `vuln_code` and `patch_code`
    that differ between the twins. Pure insertions/deletions get a 2-char anchor
    so the enclosing chunk is still marked as modified.
    """
    matcher = difflib.SequenceMatcher(None, vuln_code, patch_code, autojunk=False)
    vuln_spans: list[tuple[int, int]] = []
    patch_spans: list[tuple[int, int]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            vuln_spans.append((i1, i2))
        elif tag == "insert":
            vuln_spans.append((max(0, i1 - 1), min(len(vuln_code), i1 + 1)))

        if j2 > j1:
            patch_spans.append((j1, j2))
        elif tag == "delete":
            patch_spans.append((max(0, j1 - 1), min(len(patch_code), j1 + 1)))

    return vuln_spans, patch_spans


def tokenize_with_char_spans(
    code: str,
    tokenizer,
    chunk_size: int,
    stride: int,
    max_chunks: int,
    boundary_token_ids: set[int]
) -> list[tuple[list[int], tuple[int, int]]]:
    """
    Tokenizes `code` into boundary-snapped chunks with Head-and-Tail truncation,
    returning `(token_ids, (char_start, char_end))` per chunk.
    """
    # Reserve 1 slot at the end of every chunk for EOS (expected by CodeSage-v2)
    eos_id = tokenizer.eos_token_id
    eff_chunk_size = chunk_size - 1 if eos_id is not None else chunk_size
    eff_stride = min(stride, eff_chunk_size)

    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    all_ids: list[int] = enc["input_ids"]
    offsets: list[tuple[int, int]] = enc["offset_mapping"]

    if not all_ids:
        fallback = [eos_id] if eos_id is not None else [tokenizer.unk_token_id or 0]
        return [(fallback, (0, len(code)))]

    chunks: list[tuple[list[int], tuple[int, int]]] = []
    start = 0
    n_tokens = len(all_ids)

    while start < n_tokens:
        end = min(start + eff_chunk_size, n_tokens)

        # Snap chunk end backward to the nearest structural boundary (`\n`, `;`, `}`)
        if end < n_tokens:
            search_floor = start + max(eff_chunk_size // 2, 1)
            for idx in range(end - 1, search_floor - 1, -1):
                if all_ids[idx] in boundary_token_ids:
                    end = idx + 1
                    break
                # tok_str = tokenizer.decode([all_ids[idx]])
                # if any(ch in tok_str for ch in ("\n", ";", "}")):
                #     end = idx + 1
                #     break

        chunk_ids = all_ids[start:end] + ([eos_id] if eos_id is not None else [])
        char_span = (offsets[start][0], offsets[end - 1][1])
        chunks.append((chunk_ids, char_span))

        if end == n_tokens:
            break
        start = max(start + 1, end - (eff_chunk_size - eff_stride))

    # Head-and-Tail truncation: preserves function signature/setup + trailing cleanup
    if len(chunks) > max_chunks:
        logger.info(f"Truncating chunks from {len(chunks)} to {max_chunks} via Head-and-Tail strategy")
        half = max_chunks // 2
        chunks = chunks[:half] + chunks[-(max_chunks - half) :]

    return chunks


def build_paired_primevul_dataset(
    raw_paired_records: list[dict[str, Any]],
    tokenizer,
    chunk_size: int,
    stride: int,
    max_chunks: int,
) -> list[dict[str, Any]]:
    """
    Pairs consecutive same-commit `(target=1, target=0)` records safely via `while`
    loop (avoiding parity-shift bugs) and builds per-chunk binary diff masks.
    """
    paired_examples: list[dict[str, Any]] = []
    i = 0

    boundary_token_ids = get_code_boundary_token_ids(tokenizer)

    while i < len(raw_paired_records) - 1:
        rec_a, rec_b = raw_paired_records[i], raw_paired_records[i + 1]

        same_commit = rec_a.get("commit_id") == rec_b.get("commit_id")

        if same_commit and (rec_a["target"] != rec_b["target"]):
            vuln_rec = rec_a if rec_a["target"] == 1 else rec_b
            patch_rec = rec_b if rec_a["target"] == 1 else rec_a
            i += 2
        else:
            # Advance by 1 on an orphaned record so subsequent pairs stay aligned
            logger.warning(f"Skipping unpaired records {rec_a['idx']} and {rec_b['idx']}: commit_id={rec_a['commit_id']} vs {rec_b['commit_id']}, target={rec_a['target']} vs {rec_b['target']}")
            i += 1
            continue

        vuln_min = minify_cpp_function(vuln_rec["func"])
        patch_min = minify_cpp_function(patch_rec["func"])

        # Skip pairs whose only difference was comments or whitespace (prevents label noise)
        if vuln_min == patch_min:
            logger.warning(f"Skipping identical records {vuln_rec['idx']} and {patch_rec['idx']}")
            continue

        vuln_spans, patch_spans = get_changed_char_spans(vuln_min, patch_min)

        vuln_items = tokenize_with_char_spans(vuln_min, tokenizer, chunk_size, stride, max_chunks, boundary_token_ids)
        patch_items = tokenize_with_char_spans(patch_min, tokenizer, chunk_size, stride, max_chunks, boundary_token_ids)

        # Flag a chunk (`1.0`) if its character span overlaps any modified region
        vuln_chunks = [toks for toks, _ in vuln_items]
        vuln_diff = [
            any(s < c_end and e > c_start for s, e in vuln_spans) for _, (c_start, c_end) in vuln_items
        ]

        patch_chunks = [toks for toks, _ in patch_items]
        patch_diff = [
            any(s < c_end and e > c_start for s, e in patch_spans) for _, (c_start, c_end) in patch_items
        ]

        paired_examples.append(
            {
                "vuln_input_ids": vuln_chunks,
                "vuln_diff_mask": vuln_diff,
                "patch_input_ids": patch_chunks,
                "patch_diff_mask": patch_diff,
            }
        )

    return paired_examples