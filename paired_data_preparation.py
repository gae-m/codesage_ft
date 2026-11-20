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
    # gm: substitute with truncation around the diff spans instead of head and tail, to preserve the context of the diff
    # if len(chunks) > max_chunks:
    #     logger.info(f"Truncating chunks from {len(chunks)} to {max_chunks} via Head-and-Tail strategy")
    #     half = max_chunks // 2
    #     chunks = chunks[:half] + chunks[-(max_chunks - half) :]

    return chunks


def truncate_around_diffs(
    chunk_items: list[tuple[list[int], tuple[int, int]]],
    diff_spans: list[tuple[int, int]],
    max_chunks: int,
) -> tuple[list[list[int]], list[bool]]:
    """
    Labels each chunk (`1.0` if overlapping `diff_spans`, else `0.0`) and truncates
    bags longer than `max_chunks` by retaining chunk `0` (function signature) plus
    a symmetric context window expanding outward from the modified chunks.
    Falls back to Head-and-Tail truncation if `diff_spans` is empty.
    """

    chunks = [toks for toks, _ in chunk_items]
    # Compute binary diff label for every chunk before any truncation
    diff_flags = [
        any((c_start <= s <= c_end) or (c_start <= e <= c_end) for s, e in diff_spans) for _, (c_start, c_end) in chunk_items
    ]

    num_chunks = len(chunks)
    if num_chunks <= max_chunks:
        return chunks, diff_flags

    # Fallback for unpaired/inference functions where no diff spans exist
    if not any(diff_flags):
        half = max_chunks // 2
        return chunks[:half] + chunks[-(max_chunks - half):], diff_flags[:half] + diff_flags[-(max_chunks - half):]

    diff_indices = [idx for idx, flag in enumerate(diff_flags) if flag]

    # Edge case: massive refactoring patch where modified chunks alone exceed `max_chunks`
    if len(diff_indices) >= max_chunks:
        if 0 not in diff_indices:
            selected_list = [0] + diff_indices[: max_chunks - 1]
        else:
            selected_list = diff_indices[:max_chunks]
        kept_indices = sorted(selected_list)
        return [chunks[i] for i in kept_indices], [diff_flags[i] for i in kept_indices]

    # Normal case: pin chunk 0 (signature) + all diff chunks, then expand outward by radius `r`
    selected: set[int] = set(diff_indices)
    
    selected.add(0)

    radius = 1
    while len(selected) < max_chunks:
        added_any = False
        for center in diff_indices:
            for neighbor in (center - radius, center + radius):
                if 0 <= neighbor < num_chunks and neighbor not in selected:
                    selected.add(neighbor)
                    added_any = True
                    if len(selected) == max_chunks:
                        break
            if len(selected) == max_chunks:
                break
        if not added_any:
            break
        radius += 1

    # Sort indices so chunks stay in natural top-to-bottom code execution order
    kept_indices = sorted(selected)
    return [chunks[i] for i in kept_indices], [diff_flags[i] for i in kept_indices]


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

        vuln_items = tokenize_with_char_spans(vuln_min, tokenizer, chunk_size, stride, boundary_token_ids)
        patch_items = tokenize_with_char_spans(patch_min, tokenizer, chunk_size, stride, boundary_token_ids)


        vuln_chunks, vuln_diff = truncate_around_diffs(vuln_items, vuln_spans, max_chunks)
        patch_chunks, patch_diff = truncate_around_diffs(patch_items, patch_spans, max_chunks)

        paired_examples.append(
            {
                "vuln_input_ids": vuln_chunks,
                "vuln_diff_mask": vuln_diff,
                "patch_input_ids": patch_chunks,
                "patch_diff_mask": patch_diff,
            }
        )

    return paired_examples