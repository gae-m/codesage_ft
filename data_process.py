import jsonlines
import os


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
                selected_obj = {col: obj[col] for col in columns if col in obj}
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

    for file_name in file_names:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        select_columns(src_file, dst_file, columns=["func", "label"])
        print(f"Processed {file_name} and saved to {dst_file}")


if __name__ == "__main__":
    main()