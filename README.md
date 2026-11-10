# CodeSage-MIL: C/C++ Vulnerability Detection

## Project Summary
This project fine-tunes the CodeSage-v2-large (1.3B parameters) encoder model to detect security vulnerabilities in C/C++ code. It leverages the PrimeVul dataset, utilizing its chronological, de-duplicated splits to ensure high-fidelity evaluation. To process long, complex functions without introducing severe label noise or crashing standard GPU hardware, the pipeline introduces a custom Attention-Based Multiple Instance Learning (Attention-MIL) architecture combined with Parameter-Efficient Fine-Tuning (LoRA).

## Goal
To build a stable, VRAM-efficient, and high-performance binary classification pipeline (Safe vs. Vulnerable) that accurately identifies vulnerabilities in long C/C++ functions while strictly operating within the memory constraints of standard commercial GPUs (24GB-40GB VRAM).

## Architecture
The pipeline relies on a custom hybrid architecture:
1. **Base Encoder:** Frozen CodeSage-v2-large (loaded in `bfloat16`).
2. **Adapters:** Low-Rank Adaptation (LoRA) modules injected into all linear layers to learn vulnerability semantics.
3. **Attention-MIL Wrapper:** A custom PyTorch `nn.Module` (`AttentionMILCodeSageBCE`). Instead of predicting class probabilities per chunk, it scores the importance of multiple overlapping code chunks (a "bag") and aggregates them via a learned attention network into a single, function-level embedding.
4. **Classification Head:** A single linear layer outputting a unified logit, optimized using `BCEWithLogitsLoss`.

## Key Techniques & Mechanisms
* **Sliding Window Tokenization:** Code is sliced into overlapping 512-token chunks to capture dependencies that span beyond a single context window, capped at a hard maximum to protect VRAM.
* **Hybrid Half-Precision Training:** The frozen 1.3B base model is loaded directly into `bfloat16` to halve its memory footprint. Hugging Face's Automatic Mixed Precision (AMP) dynamically handles the FP32 mathematical stability required for the initialized custom heads during gradient updates.
* **Memory Optimization Suite:** Physical batch size is reduced to 1, while Gradient Accumulation (16 steps) and Gradient Checkpointing are employed to simulate a healthy batch size without storing massive intermediate activations.
* **Threshold-Independent Evaluation:** Model performance is primarily evaluated using AUROC, bypassing the hard `0.5` decision boundary trap that typically plagues highly imbalanced vulnerability datasets.

## Hyperparameters

### 1. Tokenization & Data Strategy
| Name | Value | Motivation |
| :--- | :--- | :--- |
| `max_length` | 512 | Balances context density and VRAM limits. Passing multiple 512-token chunks per bag prevents out-of-memory (OOM) errors. |
| `stride` | 50 | Ensures vulnerabilities spanning chunk boundaries (e.g., allocation in chunk A, unsafe free in chunk B) remain structurally intact. |
| `padding` | "max_length" | Required to pad bags with fewer chunks into uniform tensor shapes for batched processing. |
| `MAX_CHUNKS_PER_BAG`| 16 | Hardware safety cap. Restricts the sliding window to ~3,700 tokens per function, truncating massive files to prevent batch explosion and OOM crashes. |

### 2. LoRA (Parameter-Efficient Fine-Tuning)
| Name | Value | Motivation |
| :--- | :--- | :--- |
| `r` (Rank) | 16 | Provides enough parameter bandwidth to learn complex semantic vulnerability flows without bloating adapter storage size. |
| `lora_alpha` | 32 | Standard heuristic (`2 * r`) to ensure adapter weight updates have sufficient magnitude when merged with the frozen base model. |
| `lora_dropout` | 0.05 | Applies light noise to prevent the adapter from memorizing exact variable names in the PrimeVul training split. |
| `target_modules`| "all-linear" | Safely targets all dense layers (Attention Q/K/V and MLP) for superior semantic extraction without risking architecture-specific naming crashes. |

### 3. Attention-MIL Architecture
| Name | Value | Motivation |
| :--- | :--- | :--- |
| `hidden_size` | 2048 | Hardcoded architectural requirement matching the output embedding dimension of CodeSage-v2-large. |
| `attention_dim` | 256 | Acts as a regularization bottleneck. Forces the attention network to learn broad patterns of "danger" across the chunks rather than memorizing exact 2048-dim vectors. |

### 4. Training Arguments
| Name | Value | Motivation |
| :--- | :--- | :--- |
| `num_train_epochs` | 5 | Restricts training duration to prevent the model from overfitting on the imbalanced training split. |
| `per_device_train_batch_size` | 1 | A logical batch of 1 "bag" processes up to 16 sequences simultaneously. Anything higher exceeds VRAM limits. |
| `gradient_accumulation_steps` | 16 | Reconstructs a stable effective batch size of 16 (1 * 16), which is mathematically critical for updating gradients on highly imbalanced data. |
| `learning_rate` | 2e-4 | Required magnitude for LoRA. Training only ~1% of the network requires a higher learning rate than full model fine-tuning (2e-5). |
| `warmup_ratio` | 0.1 | Linearly increases learning rate over the first 10% of steps. Prevents the randomly initialized custom heads from immediately destabilizing the frozen network. |
| `weight_decay` | 0.01 | Standard AdamW regularization to penalize large weights and improve zero-day generalization. |
| `max_grad_norm` | 1.0 | Gradient clipping. Prevents exploding gradients if the Attention-MIL network assigns extreme weights early in training. |
| `gradient_checkpointing`| True | Trades a ~20% increase in compute time for a ~50% reduction in VRAM by recalculating backward-pass activations dynamically. |
| `bf16` | True | Bfloat16 provides a massive dynamic range, completely eliminating `NaN` loss scaling errors common when fine-tuning code models in FP16. |
| `save_strategy` | "no" | Disables native Hugging Face checkpointing to avoid dumping the frozen 1.3B base model to disk, relying entirely on the custom saving subclass. |

## Resuls
TBD