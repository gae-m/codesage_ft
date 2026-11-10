# Scientific References: CodeSage-MIL Vulnerability Detection

This document lists the foundational scientific research and papers that inspired the architectural decisions, optimization techniques, and mechanisms implemented in this machine learning project.

### 1. Dataset & Benchmark

* **PrimeVul Dataset:**

  * **Paper:** *Vulnerability Detection with Code Language Models: How Far Are We?*

  * **Authors:** Yizhan Ding, Zeyu Wang, Wasi Uddin Ahmad, Hieu Ding, Murali Krishna Ramanathan, et al.

  * **Publication:** arXiv 2024

  * **Link / ArXiv:** [arXiv:2403.18624](https://arxiv.org/abs/2403.18624)

  * **Relevance:** Introduces the PrimeVul benchmark dataset, providing high-quality, de-duplicated C/C++ vulnerability samples with realistic temporal train/validation/test splits to mitigate data leakage.

### 2. Parameter-Efficient Fine-Tuning

* **LoRA (Low-Rank Adaptation):**

  * **Paper:** *LoRA: Low-Rank Adaptation of Large Language Models*

  * **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.

  * **Publication:** ICLR 2022

  * **Link / ArXiv:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

  * **Relevance:** Foundational method for injecting rank-decomposition matrices into Transformer layers to freeze base weights and save VRAM.

### 3. Multiple Instance Learning & Attention

* **Gated Attention-Based MIL:**

  * **Paper:** *Attention-based Deep Multiple Instance Learning*

  * **Authors:** Maximilian Ilse, Jakub M. Tomczak, Max Welling.

  * **Publication:** ICML 2018

  * **Link / ArXiv:** [arXiv:1802.04712](https://arxiv.org/abs/1802.04712)

  * **Relevance:** Introduced the learnable Attention-MIL framework and the specific $\tanh$ and sigmoid gated attention mechanism used to score overlapping code chunks.

### 4. Precision & Numerical Stability

* **Mixed Precision Training:**

  * **Paper:** *Mixed Precision Training*

  * **Authors:** Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Evgeny Borisov, Peter Hawkins, Hao Yin, et al.

  * **Publication:** ICLR 2018

  * **Link / ArXiv:** [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

  * **Relevance:** Established the mathematical viability of mixing 16-bit and 32-bit floats during neural network training to preserve accuracy while accelerating compute.

* **Bfloat16 (Brain Floating Point):**

  * **Paper:** *A Study of BFLOAT16 for Deep Learning Training*

  * **Authors:** Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, et al.

  * **Publication:** arXiv 2019

  * **Link / ArXiv:** [arXiv:1905.12322](https://arxiv.org/abs/1905.12322)

  * **Relevance:** Proved the superiority of the BF16 format (8 exponent bits) over FP16 for preventing overflow/underflow in complex language models.

### 5. Memory Optimization & Scaling

* **Gradient Checkpointing:**

  * **Paper:** *Training Deep Nets with Sublinear Memory Cost*

  * **Authors:** Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin.

  * **Publication:** arXiv 2016

  * **Link / ArXiv:** [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)

  * **Relevance:** The core mechanism utilized to discard and recompute intermediate forward-pass activations, keeping the MIL architecture's VRAM usage within consumer GPU limits.

* **Gradient Accumulation & Large Batch Training:**

  * **Paper:** *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*

  * **Authors:** Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He.

  * **Publication:** arXiv 2017

  * **Link / ArXiv:** [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

  * **Relevance:** While gradient accumulation is a programmatic implementation trick, this paper provides the foundational theoretical rules (like the Linear Scaling Rule) for maintaining gradient stability and matching statistics when simulating large effective batches over multiple smaller forward passes.

### 6. Architectural Components

* **GELU Activation Function:**

  * **Paper:** *Gaussian Error Linear Units (GELUs)*

  * **Authors:** Dan Hendrycks, Kevin Gimpel.

  * **Publication:** arXiv 2016

  * **Link / ArXiv:** [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)

  * **Relevance:** The non-linear activation function powering the custom 2-Layer MLP classification head, now standard in modern Transformer pipelines.

### 7. Base Model & Pre-training

* **CodeSage Encoder:**

  * **Paper:** *Code Representation Learning At Scale*

  * **Authors:** Dejiao Zhang, Wasi Uddin Ahmad, Ming Li, Shrikant Shenoy, et al. (Amazon Web Services).

  * **Publication:** ICLR 2024

  * **Link / ArXiv:** [arXiv:2402.01935](https://arxiv.org/abs/2402.01935)

  * **Relevance:** Introduces the CodeSage model family. It details the bidirectional masked language modeling approach and the massive pre-training pipeline that allows the model to act as a highly effective baseline for C/C++ structural embeddings.