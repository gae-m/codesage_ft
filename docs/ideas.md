# Notes


## Problems
The training loss seems stucked on a plateau of 0.7. Codesage model does not support gradient checkpointing. To not cause OOM error with 32GB of VRAM we have setted MAX_CHUNKS_PER_BAG=8. A lot of functions in the training set exceed this value, causing a truncation of them. We have to mititgate this problem.  


## Ideas
1. Data augmentation: adding dead-code, variable names normalization, ...
2. Input normalization: remove empty lines, comments, ... (can be useless if the tokenizer already managed them)
3. Exploit the Primevul paired dataset to implment some sort of contrastive learning for the fine-tuning