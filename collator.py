import torch
from typing import Any

class AttentionMILCollator:
    def __init__(self, pad_token_id: int, empty_chunk_ids: list[int]):
        self.pad_token_id = pad_token_id
        self.empty_chunk_ids = empty_chunk_ids

    # def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
    #     # Find the maximum number of chunks in this specific batch
    #     max_chunks = max(len(item['input_ids']) for item in batch)
    #     seq_len = len(batch[0]['input_ids'][0])
        
    #     batch_input_ids, batch_attention_mask, batch_chunk_mask, labels = [], [], [], []
        
    #     for item in batch:
            
    #         num_chunks = len(item['input_ids'])
            
    #         # chunk_mask tells the attention mechanism which chunks are real vs. padding
    #         chunk_mask = [1] * num_chunks + [0] * (max_chunks - num_chunks)
            
    #         # Pad the bag with dummy sequences if it has fewer chunks than the max in the batch
    #         dummy_sequence = [self.pad_token_id] * seq_len
    #         dummy_attention = [0] * seq_len
            
    #         padded_input_ids = item['input_ids'] + [dummy_sequence] * (max_chunks - num_chunks)
    #         padded_attention_mask = item['attention_mask'] + [dummy_attention] * (max_chunks - num_chunks)
            
    #         batch_input_ids.append(padded_input_ids)
    #         batch_attention_mask.append(padded_attention_mask)
    #         batch_chunk_mask.append(chunk_mask)
    #         labels.append(item['label'])
    #     return {
    #         "input_ids": torch.tensor(batch_input_ids),
    #         "attention_mask": torch.tensor(batch_attention_mask),
    #         "chunk_mask": torch.tensor(batch_chunk_mask),
    #         "labels": torch.tensor(labels)
    #     }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Dynamically pads both the number of chunks per bag (dim 1) and the variable
        boundary-snapped sequence length per chunk (dim 2).
        """
        max_chunks = max(len(f["input_ids"]) for f in batch)
        raw_max_seq = max(len(chunk) for f in batch for chunk in f["input_ids"])
        # Pad sequence dimension to a multiple of 8 for Tensor Core alignment
        max_seq_len = ((raw_max_seq + 7) // 8) * 8

        # Create a dummy chunk with padding
        len_empty_chunk = len(self.empty_chunk_ids)
        empty_chunk_ids = self.empty_chunk_ids + [self.pad_token_id] * (max_seq_len - len_empty_chunk)
        empty_chunk_mask = len_empty_chunk * [1] + [0] * (max_seq_len - len_empty_chunk)

        batch_input_ids = []
        batch_attention_mask = []
        batch_chunk_mask = []
        batch_labels = []

        for f in batch:
            padded_ids = []
            padded_masks = []
            for ids in f["input_ids"]:
                pad_len = max_seq_len - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                padded_masks.append([1] * len(ids) + [0] * pad_len)

            n_real = len(padded_ids)
            n_pad_chunks = max_chunks - n_real
            padded_ids.extend([empty_chunk_ids] * n_pad_chunks)
            padded_masks.extend([empty_chunk_mask] * n_pad_chunks)

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_masks)
            batch_chunk_mask.append([1] * n_real + [0] * n_pad_chunks)
            batch_labels.append(f["label"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.bool),
            "chunk_mask": torch.tensor(batch_chunk_mask, dtype=torch.bool),
            "labels": torch.tensor(batch_labels, dtype=torch.bool),
        }