import os
from typing import Any, cast

import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset, DatasetDict
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from time import time

from collator import AttentionMILCollator
from classifier_wrapper import AttentionMILCodeSage


ENCODER = "codesage/codesage-large-v2"
MAX_CHUNKS_PER_BAG = 8  # Limit the number of chunks per function to avoid memory issues
# Define the mapping of your splits to the local file paths
DATAPATHS = {
    "train": r"data\primevul\primevul_train_paired.jsonl",
    "validation": r"data\primevul\primevul_valid_paired.jsonl",
    "test": r"data\primevul\primevul_valid_paired.jsonl"
}

def load_data(data_files: dict[str, str], tokenizer: AutoTokenizer) -> DatasetDict:
    # Load the dataset using the 'json' builder
    raw_dataset: DatasetDict = load_dataset("json", data_files=data_files)
    print(raw_dataset)

    def tokenize_into_bags(example: dict[str, Any]) -> dict[str, Any]:
        # Tokenize a single function into multiple overlapping chunks
        tokenized = tokenizer(
            example["func"],  # Assuming the code is stored under the key "func"
            truncation=True,
            max_length=512,
            stride=50,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length"
        )
        
        # tokenized["input_ids"] is now a list of lists. Shape: (num_chunks, 512)
        input_ids = tokenized["input_ids"][:MAX_CHUNKS_PER_BAG]
        attention_mask = tokenized["attention_mask"][:MAX_CHUNKS_PER_BAG]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": example["target"]  # Assuming the label is stored under the key "target"
        }

    # Apply without batched=True to keep chunks grouped by their parent function
    tokenized_dataset = raw_dataset.map(tokenize_into_bags, remove_columns=["code"])
    return tokenized_dataset


def compute_auroc_metrics(eval_pred):
    logits, labels = eval_pred
    
    # 1. Apply sigmoid to convert raw logits into probabilities
    probabilities = expit(logits, axis=1)
    
    # 2. Extract the probability of the positive class (Class 1: Vulnerable)
    positive_class_probs = probabilities[:, 1]
    
    # 3. Calculate AUROC
    try:
        auroc = roc_auc_score(labels, positive_class_probs)
    except ValueError:
        # Handles edge cases where a batch might only contain one class during testing
        auroc = 0.5 
        
    return {"val/auroc": auroc}


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
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

    tokenized_dataset = load_data(data_files, tokenizer)

    if dbg:
        print("Debug mode: Reducing dataset size for quick testing...")
        train_set = tokenized_dataset["train"].select(range(min(10, len(tokenized_dataset["train"]))))
        eval_set = tokenized_dataset["validation"].select(range(min(10, len(tokenized_dataset["validation"]))))
        test_set = tokenized_dataset["test"].select(range(min(10, len(tokenized_dataset["test"]))))
        tokenized_dataset = DatasetDict({"train": train_set, "validation": eval_set, "test": test_set})

    # Load base CodeSage without the classification head
    base_encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

    # Apply LoRA to the base encoder
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"], 
        # Do NOT include "classifier" here, because our classifier is in the wrapper, not the PEFT model
    )
    peft_encoder = get_peft_model(base_encoder, lora_config)

    # Initialize the Attention-MIL model
    mil_model = AttentionMILCodeSage(encoder=peft_encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mil_model = mil_model.to(device) # Move to GPU

    # Create the Trainer
    training_args = TrainingArguments(
        output_dir=out_dir,
        max_steps=3 if dbg else -1,
        per_device_train_batch_size=2, # Keep batch size small; you are processing (batch * chunks) sequences at once
        learning_rate=2e-4,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=4, # Simulate a larger batch size
        num_train_epochs=5,
        bf16=True,
        metric_for_best_model="val/auroc",
        logging_steps=50,
        logging_first_step=True,
        report_to=["tensorboard"],
        logging_dir=out_dir + "/logs",
        # Crucial: Since our model is custom and doesn't subclass PreTrainedModel cleanly for saving, 
        # disable standard HF saving and save state_dict manually, or handle carefully.
        save_strategy="no" 
    )

    # While CodeSage does have a pad token, some code tokenizers (like GPT-2 base) do not have one set by default.
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=mil_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=AttentionMILCollator(pad_token_id=tokenizer.pad_token_id),
        compute_metrics=compute_auroc_metrics
    )

    trainer.train()

    # 2. Evaluate on the Test Set
    print("Running final evaluation on the Test Set...")
    test_results = trainer.predict(tokenized_dataset["test"])

    # 3. Print the comprehensive metrics
    for metric_name, value in test_results.metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Save only the adapters and the classifier head
    save_model(mil_model, out_dir)



def load_model(model_dir: str, encoder_name: str = "codesage/codesage-large-v2") -> AttentionMILCodeSage:
    # 1. Load the base CodeSage model
    base_model = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

    # 2. Load the trained LoRA adapters onto the base model
    peft_encoder = PeftModel.from_pretrained(base_model, os.path.join(model_dir, "lora-adapters"))

    # 3. Initialize your custom architecture wrapper
    loaded_mil_model = AttentionMILCodeSage(
        encoder=peft_encoder, 
        hidden_size=2048, 
        attention_dim=256, 
        num_classes=2
    )

    # 4. Load the custom attention and classification heads
    custom_weights = torch.load(os.path.join(model_dir, "custom_heads.pt"))
    loaded_mil_model.attention.load_state_dict(custom_weights["attention_state"])
    loaded_mil_model.classifier.load_state_dict(custom_weights["classifier_state"])

    # Set to evaluation mode
    loaded_mil_model.eval()

    return loaded_mil_model


enc_name_map = {
    "codesage/codesage-large-v2": "cs2l",
    "codesage/codesage-base-v2": "cs2b",
    "codesage/codesage-small-v2": "cs2s"
}


def main():
    # Define the output directory for saving the model
    experiment_name = f"{enc_name_map[ENCODER]}_mil_ft_{int(time())}"
    out_dir = f"out/{experiment_name}"

    # Fine-tune the model
    fine_tuning(DATAPATHS, out_dir, encoder_name=ENCODER)
