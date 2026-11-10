import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

class AttentionMILCodeSage(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size=2048, attention_dim=256, num_classes=2):
        super().__init__()
        self.encoder = encoder
        
        # The Attention Network: Learns to score the importance of each chunk
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, chunk_mask=None, labels=None):
        # Shapes: input_ids -> (batch_size, max_chunks, seq_len)
        batch_size, max_chunks, seq_len = input_ids.size()
        
        # 1. Flatten the batch to pass all chunks through CodeSage simultaneously
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        # Extract the [CLS] embedding for every chunk
        chunk_embeddings = outputs.last_hidden_state[:, 0, :] 
        
        # 2. Reshape back into bags -> (batch_size, max_chunks, hidden_size)
        bag_embeddings = chunk_embeddings.view(batch_size, max_chunks, -1)
        
        # 3. Calculate Attention Scores -> (batch_size, max_chunks)
        attn_scores = self.attention(bag_embeddings).squeeze(-1) 
        
        # Mask out the padding chunks so they receive 0 attention
        if chunk_mask is not None:
            attn_scores = attn_scores.masked_fill(chunk_mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=1) 
        
        # 4. Aggregate chunks into a single function-level embedding
        # bmm: (batch, 1, max_chunks) x (batch, max_chunks, hidden_size) -> (batch, 1, hidden_size)
        function_representation = torch.bmm(attn_weights.unsqueeze(1), bag_embeddings).squeeze(1)
        
        # 5. Classify the entire function
        logits = self.classifier(function_representation)
        
        # Compute loss natively so Hugging Face Trainer can use it seamlessly
        loss = None
        if labels is not None:
            # You can replace this with Focal Loss
            loss_fct = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits, "attention_weights": attn_weights}