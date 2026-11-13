import re

# Master C/C++ lexer regex: matches literals BEFORE comments so strings containing
# "//" or "/* */" are never accidentally corrupted.
CPP_LEXER_RE = re.compile(
    r'(?P<raw_str>R"([^\s()\\]{0,16})\([\s\S]*?\)\2")'   # C++11 raw string R"delim(...)delim"
    r'|(?P<str>"(?:\\[\s\S]|[^"\\])*")'                  # Standard double-quoted string
    r"|(?P<char>'(?:\\[\s\S]|[^'\\])*')"                 # Single-quoted character literal
    r'|(?P<line_comment>//(?:\\(?:\r?\n)|[^\r\n])*)'     # Single-line // comment (with \ splicing)
    r'|(?P<block_comment>/\*[\s\S]*?\*/)',               # Multi-line /* ... */ comment
    re.MULTILINE,
)

# Operator pairs that would change meaning or become comments if adjacent whitespace is removed
DANGEROUS_PAIRS: set[tuple[str, str]] = {
    ("/", "*"), ("*", "/"), ("/", "/"),
    ("+", "+"), ("-", "-"), ("<", "<"), (">", ">"),
    ("&", "&"), ("|", "|"), ("=", "="), ("!", "="),
    ("<", "="), (">", "="), ("+", "="), ("-", "="),
    ("*", "="), ("/", "="), ("%", "="), ("&", "="),
    ("|", "="), ("^", "="), ("-", ">"), (":", ":"),
    (".", "*"),
}

# Preserve C/C++ keywords and security-critical standard/POSIX APIs during identifier compression
PRESERVED_TOKENS: set[str] = {
    # C/C++ keywords & primitive types
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
    "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct",
    "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "bool",
    "class", "delete", "false", "namespace", "new", "nullptr", "operator", "private",
    "protected", "public", "template", "this", "throw", "true", "try", "catch", "using",
    "virtual", "size_t", "ssize_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "NULL", "EOF",
    # High-signal vulnerability sinks & memory APIs
    "malloc", "calloc", "realloc", "free", "alloca", "memcpy", "memmove", "memset",
    "memcmp", "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp", "strlen",
    "strnlen", "sprintf", "snprintf", "vsprintf", "vsnprintf", "printf", "fprintf",
    "scanf", "sscanf", "fscanf", "gets", "fgets", "fread", "fwrite", "read", "write",
    "recv", "recvfrom", "send", "sendto", "socket", "ioctl", "mmap", "munmap",
    "open", "close", "fopen", "fclose", "system", "execve", "popen", "assert",
}


def strip_comments_safe(code: str) -> str:
    """Technique 1: Removes // and /* */ comments while preserving string/char literals."""
    def _replacer(match: re.Match) -> str:
        if match.group("line_comment") or match.group("block_comment"):
            # Replace with a single space so `int/*c*/x` becomes `int x`, not `intx`
            return " "
        return match.group(0)

    return CPP_LEXER_RE.sub(_replacer, code)


def strip_if_zero_blocks(code: str) -> str:
    """Technique 2: Removes `#if 0 ... #endif` dead blocks while tracking nested `#if` and `#else`."""
    lines = code.splitlines(keepends=True)
    out = []
    stack = []  # Tracks "skip" (#if 0), "keep_else" (#else of #if 0), or "normal"

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            directive = re.sub(r"^#\s*", "", stripped)
            if re.match(r"^if\s+0\b", directive) and "skip" not in stack:
                stack.append("skip")
                continue
            elif re.match(r"^(if|ifdef|ifndef)\b", directive):
                stack.append("skip" if "skip" in stack else "normal")
                if "skip" not in stack:
                    out.append(line)
                continue
            elif re.match(r"^(else|elif)\b", directive) and stack:
                if stack[-1] == "skip" and stack.count("skip") == 1:
                    stack[-1] = "keep_else"
                    if directive.startswith("elif"):
                        out.append(re.sub(r"^(\s*#\s*)elif\b", r"\1if", line))
                    continue
                elif "skip" not in stack:
                    out.append(line)
                continue
            elif re.match(r"^endif\b", directive) and stack:
                state = stack.pop()
                if state == "normal" and "skip" not in stack:
                    out.append(line)
                continue

        if "skip" not in stack:
            out.append(line)

    return "".join(out)


def compress_long_identifiers(code: str, min_len: int = 8) -> str:
    """
    Technique 5 (Optional / Lossy): Renames non-API identifiers of length >= `min_len`
    to compact `v0, v1, ...` symbols while preserving literals and standard library APIs.
    """
    mapping: dict[str, str] = {}
    counter = 0

    def _rename_outside_literals(segment: str) -> str:
        nonlocal counter
        def _id_sub(m: re.Match) -> str:
            nonlocal counter
            ident = m.group(0)
            if ident in PRESERVED_TOKENS or len(ident) < min_len:
                return ident
            if ident not in mapping:
                mapping[ident] = f"v{counter}"
                counter += 1
            return mapping[ident]

        return re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", _id_sub, segment)

    out = []
    last_idx = 0
    for m in CPP_LEXER_RE.finditer(code):
        if m.start() > last_idx:
            out.append(_rename_outside_literals(code[last_idx:m.start()]))
        out.append(m.group(0))  # Keep string/char literals untouched
        last_idx = m.end()
    if last_idx < len(code):
        out.append(_rename_outside_literals(code[last_idx:]))

    return "".join(out)


def minify_cpp_function(
    code: str,
    aggressive_whitespace: bool = False,
    rename_long_identifiers: bool = False,
) -> str:
    """
    Full C/C++ minification pipeline:
      1. Strips comments safely.
      2. Strips `#if 0` dead preprocessor blocks.
      3. Strips indentation, blank lines, and redundant horizontal whitespace.
      4. Optionally packs lines/delimiters (`aggressive_whitespace=True`).
      5. Optionally shortens long local identifiers (`rename_long_identifiers=True`).
    """
    code = strip_comments_safe(code)
    code = strip_if_zero_blocks(code)

    if rename_long_identifiers:
        code = compress_long_identifiers(code)

    # Split into literal vs. non-literal spans so whitespace inside strings is never altered
    parts = []
    last_idx = 0
    for m in CPP_LEXER_RE.finditer(code):
        if m.start() > last_idx:
            parts.append((False, code[last_idx:m.start()]))
        parts.append((True, m.group(0)))
        last_idx = m.end()
    if last_idx < len(code):
        parts.append((False, code[last_idx:]))

    normalized = []
    for is_literal, text in parts:
        if is_literal:
            normalized.append(text)
        else:
            text = re.sub(r"[ \t]+", " ", text)
            if aggressive_whitespace:
                text = re.sub(r" ?([{}()\[\];,]) ?", r"\1", text)
            normalized.append(text)

    cleaned_lines = [line.strip() for line in "".join(normalized).splitlines() if line.strip()]

    # Mode A: Standard Minification (preserves line structure, removes indentation & blank lines)
    if not aggressive_whitespace:
        return "\n".join(cleaned_lines)

    # Mode B: Aggressive Single-Line Packing (keeps newlines only where preprocessor requires them)
    out = []
    in_pp = False
    for line in cleaned_lines:
        is_pp = line.startswith("#") or in_pp
        in_pp = is_pp and line.endswith("\\")
        if is_pp:
            out.append("\n" + line + "\n")
        else:
            if out and not out[-1].endswith("\n"):
                prev_ch, curr_ch = out[-1][-1], line[0]
                needs_space = (
                    ((prev_ch.isalnum() or prev_ch == "_") and (curr_ch.isalnum() or curr_ch == "_"))
                    or (prev_ch, curr_ch) in DANGEROUS_PAIRS
                )
                out.append((" " if needs_space else "") + line)
            else:
                out.append(line)

    return "".join(out).strip()


import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def benchmark_minification(code_funcs: list, tokenizer, n_samples=-1) -> None:
    orig_lens, std_lens, agg_lens = [], [], []
    code_funcs = code_funcs[:n_samples]
    for raw in code_funcs:
        std = minify_cpp_function(raw, aggressive_whitespace=False)
        agg = minify_cpp_function(raw, aggressive_whitespace=True)

        orig_lens.append(len(tokenizer.encode(raw, add_special_tokens=False)))
        std_lens.append(len(tokenizer.encode(std, add_special_tokens=False)))
        agg_lens.append(len(tokenizer.encode(agg, add_special_tokens=False)))

    print(f"Original mean tokens:   {np.mean(orig_lens):.1f}, 95th pct: {np.percentile(orig_lens, 95):.0f}")
    print(f"Standard minification:  {np.mean(std_lens):.1f} (-{(1 - np.mean(std_lens)/np.mean(orig_lens))*100:.1f}%), 95th pct: {np.percentile(std_lens, 95):.0f}")
    print(f"Aggressive minification:{np.mean(agg_lens):.1f} (-{(1 - np.mean(agg_lens)/np.mean(orig_lens))*100:.1f}%), 95th pct: {np.percentile(agg_lens, 95):.0f}")


def main():
    data_files = {
        "train": r"data/processed/primevul/primevul_train_paired.jsonl",
        "validation": r"data/processed/primevul/primevul_valid_paired.jsonl",
        "test": r"data/processed/primevul/primevul_test_paired.jsonl"
    }
    encoder_name = r"codesage/codesage-large-v2"

    dataset: DatasetDict = load_dataset("json", data_files=data_files) #type: ignore

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

    code_funcs = dataset["train"]["func"]
    benchmark_minification(code_funcs, tokenizer)


if __name__ == "__main__":
    main()

