# TODO: Remember: the training dataset must be paired, while the test dataset not.
import argparse
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig, get_peft_model
import torch
from time import perf_counter, time
from dataclasses import dataclass, asdict
from finetune import encoder_name_map
from paired_data_preparation import build_paired_primevul_dataset
from data_process import prepare_mil_dataset

torch.set_float32_matmul_precision('high')

logger = logging.getLogger("codesage_ft")


data_files = {
    "train": r"data/processed/primevul/primevul_train_paired.jsonl",
    "valid": r"data/processed/primevul/primevul_valid_paired.jsonl",
    "test": r"data/processed/primevul/primevul_test_paired.jsonl"
}

@dataclass
class FtOpts:
    encoder = "codesage/codesage-large-v2"
    debug = False
    max_chunks_per_bag = 16
    max_seq_len = 2048
    stride_ratio = 0.75
    dataloader_workers = 4
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lr = 2e-4
    lr_warmup_ratio = 0.1
    epochs = 5
    training_batch_size = 1
    validation_batch_size = 1
    gradient_accumulation_steps = 8
    checkpointing_ratio = 0.5


def set_logging(log_file_path: str):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file_path)):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)



def paired_finetuning(data_files: dict[str, str], opts: FtOpts, out_dir: str):
    logger.info("Parameters...\n" + "\n".join(f"{k}: {v}" for k, v in asdict(opts).items()))

    logger.info("Loading and tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(opts.encoder, trust_remote_code=True)

    raw_dataset: DatasetDict = load_dataset("json", data_files=data_files) # type: ignore

    raw_train_set = concatenate_datasets([raw_dataset["train"], raw_dataset["valid"]])
    raw_test_set = raw_dataset["test"]

    tokenized_train_set = build_paired_primevul_dataset(
        raw_paired_records=raw_train_set,
        tokenizer=tokenizer,
        chunk_size=opts.max_seq_len,
        stride=int(opts.max_seq_len * opts.stride_ratio),
        max_chunks=opts.max_chunks_per_bag
    )
    
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
    logger.info("Tokenization complete. Dataset overview:")
    logger.info(tokenized_dataset)

    if opts.debug:
        logger.info("Debug mode: Reducing dataset size for quick testing...")
        train_set = tokenized_dataset["train"].select(range(min(10, len(tokenized_dataset["train"]))))
        eval_set = tokenized_dataset["validation"].select(range(min(10, len(tokenized_dataset["validation"]))))
        test_set = tokenized_dataset["test"].select(range(min(10, len(tokenized_dataset["test"]))))
        tokenized_dataset = DatasetDict({"train": train_set, "validation": eval_set, "test": test_set})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_support = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    logger.info("Using device:", device)
    logger.info("bf16 support:", bf16_support)

    logger.info("Loading the model...")
    config = AutoConfig.from_pretrained(opts.encoder, trust_remote_code=True)
    config.attention_dropout_prob = 0.0
    config.residual_dropout_prob = 0.0
    config.embedding_dropout_prob = 0.0

    # Load base CodeSage without the classification head
    base_encoder = AutoModel.from_pretrained(
        opts.encoder,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16 if bf16_support else torch.float32
        # Loads the frozen weights directly in 16-bit
    )

    # Apply LoRA to the base encoder
    lora_config = LoraConfig(
        r=opts.lora_r,
        lora_alpha=opts.lora_alpha,
        lora_dropout=opts.lora_dropout,
        target_modules="all-linear",  # Apply LoRA to all linear layers in the encoder
        fan_in_fan_out=True,
        bias="none"
        # Do NOT include "classifier" here, because our classifier is in the wrapper, not the PEFT model
    )

    peft_encoder = get_peft_model(base_encoder, lora_config)

    # Initialize the Attention-MIL model
    mil_model = AttentionMILCodeSage(encoder=peft_encoder, hidden_size=base_encoder.config.hidden_size)

    logger.info("Model loaded. Preparing for training...")

    mil_model = mil_model.to(device) # Move to GPU

    # Create the Trainer
    training_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=3 if opts.debug else -1,
        group_by_length=True,                             # Useful only if batch_size > 1, like for the eval phase in this case
        per_device_train_batch_size=opts.training_batch_size,  # Keep batch size small; you are processing (batch * chunks) sequences at once
        per_device_eval_batch_size=opts.validation_batch_size,                     # You can increase this if you have enough GPU memory
        learning_rate=opts.lr,
        gradient_accumulation_steps=opts.gradient_accumulation_steps,   # Simulate a larger batch size
        gradient_checkpointing=True,                      # Save memory by not storing intermediate activations. CodeSage does not support gradient checkpointing natively.
        gradient_checkpointing_kwargs={"checkpoint_ratio": opts.checkpointing_ratio}, # Checkpoint only CHECKPOINTING_RATIO of the CodeSage blocks to balance memory and speed
        warmup_ratio=opts.lr_warmup_ratio,
        num_train_epochs=opts.epochs,
        bf16=bf16_support,
        # Logging and evaluation strategies
        logging_steps=64,
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
        dataloader_num_workers=opts.dataloader_workers,
        dataloader_persistent_workers=True,
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

    logger.info("Starting training...")
    start_time = perf_counter()
    try:
        trainer.train(
            ignore_keys_for_eval=["attention_weights"]  # Ignore the attention weights during evaluation to avoid unnecessary computation
        )
        end_time = perf_counter()
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")
    except KeyboardInterrupt:
        end_time = perf_counter()
        logger.info(f"Training interrupted by user after {end_time - start_time:.2f} seconds.")

    # 2. Evaluate on the Test Set
    logger.info("Running final evaluation on the Test Set...")
    test_results = trainer.evaluate(tokenized_dataset["test"], ignore_keys=["attention_weights"]) # type: ignore

    # 3. Print the comprehensive metrics
    for metric_name, value in test_results.items():
        logger.info(f"{metric_name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeSage with Attention-MIL on Paired Datasets")
    parser.add_argument("--encoder", "-e", type=str, help="Pre-trained CodeSage encoder to use.")
    parser.add_argument("--dbg", "-d", action="store_true", help="Enable debug mode for quick testing.")

    args = parser.parse_args()

    opts = FtOpts()
    if args.encoder:
        opts.encoder = args.encoder
    if args.dbg:
        opts.debug = True

    # Define the output directory for saving the model
    experiment_name = f"{encoder_name_map[opts.encoder]}_mil_paired_ft_{int(time())}"
    if args.dbg:
        experiment_name = 'dbg_' + experiment_name

    out_dir = f"out/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Fine-tune the model
    paired_finetuning(data_files, opts, out_dir)


if __name__ == "__main__":
    main()