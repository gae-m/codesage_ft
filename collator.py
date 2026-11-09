import torch

class AttentionMILCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([item['label'] for item in batch])
        
        # Find the maximum number of chunks in this specific batch
        max_chunks = max(len(item['input_ids']) for item in batch)
        seq_len = len(batch[0]['input_ids'][0]) # Always 512
        
        batch_input_ids, batch_attention_mask, batch_chunk_mask = [], [], []
        
        for item in batch:
            num_chunks = len(item['input_ids'])
            
            # chunk_mask tells the attention mechanism which chunks are real vs. padding
            chunk_mask = [1] * num_chunks + [0] * (max_chunks - num_chunks)
            
            # Pad the bag with dummy sequences if it has fewer chunks than the max in the batch
            dummy_sequence = [self.pad_token_id] * seq_len
            dummy_attention = [0] * seq_len
            
            padded_input_ids = item['input_ids'] + [dummy_sequence] * (max_chunks - num_chunks)
            padded_attention_mask = item['attention_mask'] + [dummy_attention] * (max_chunks - num_chunks)
            
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_chunk_mask.append(chunk_mask)
            
        return {
            "input_ids": torch.tensor(batch_input_ids),
            "attention_mask": torch.tensor(batch_attention_mask),
            "chunk_mask": torch.tensor(batch_chunk_mask),
            "labels": labels
        }