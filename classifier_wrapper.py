import torch
import torch.nn as nn

from cs_checkpointing import disable_codesage_block_checkpointing, enable_codesage_selective_checkpointing


class GatedAttention(nn.Module):
    """
    A standalone Gated Attention module for Multiple Instance Learning.
    Computes attention scores using a gated mechanism (tanh * sigmoid) 
    and aggregates the bag of chunks into a single representation.
    """
    def __init__(self, hidden_size, attention_dim, dropout_prob=0.1):
        super().__init__()
        # Feature extractor (V)
        self.attention_v = nn.Linear(hidden_size, attention_dim)
        # Gate mechanism (U)
        self.attention_u = nn.Linear(hidden_size, attention_dim)
        # Regularization
        self.attention_dropout = nn.Dropout(dropout_prob)
        # Final scorer
        self.attention_weights = nn.Linear(attention_dim, 1)

    def forward(self, bag_embeddings, chunk_mask=None):
        # bag_embeddings shape: (batch_size, max_chunks, hidden_size)
        
        # 1. Feature extraction and gating
        v = torch.tanh(self.attention_v(bag_embeddings))
        u = torch.sigmoid(self.attention_u(bag_embeddings))
        
        # 2. Element-wise multiplication to apply the gate
        gated_features = v * u 
        gated_features = self.attention_dropout(gated_features)
        
        # 3. Project down to 1 score per chunk
        attn_scores = self.attention_weights(gated_features).squeeze(-1) # (batch_size, max_chunks)
        
        # 4. Mask out padded chunks so they receive -inf score (0 attention)
        if chunk_mask is not None:
            attn_scores = attn_scores.masked_fill(chunk_mask == 0, -1e9)
            
        # 5. Convert scores to probabilities
        attn_weights = torch.softmax(attn_scores, dim=1) # (batch_size, max_chunks)
        
        # 6. Aggregate chunks into a single function-level embedding
        # bmm: (batch, 1, max_chunks) x (batch, max_chunks, hidden_size) -> (batch, 1, hidden_size)
        function_representation = torch.bmm(attn_weights.unsqueeze(1), bag_embeddings).squeeze(1)
        
        return function_representation, attn_weights


class AttentionMILCodeSage(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, attention_dim=256, mlp_dropout=0.1):
        super().__init__()
        self.encoder = encoder
        
        # The Attention Network: Learns to score the importance of each chunk
        # self.attention = nn.Sequential(
        #     nn.Linear(hidden_size, attention_dim),
        #     nn.Tanh(),
        #     nn.Linear(attention_dim, 1)
        # )
        self.attention = GatedAttention(hidden_size, attention_dim)

        mlp_hidden_dim = hidden_size // 4  # You can adjust this based on your needs
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim), 
            nn.GELU(),                            # Non-linear activation (GELU is standard for LLMs)
            nn.Dropout(mlp_dropout),              # dropout to prevent overfitting on imbalanced data
            nn.Linear(mlp_hidden_dim, 1)          # Final projection to 1 logit for BCE
        )

        # You can replace this with Focal Loss
        self.loss_fct = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # enable_codesage_block_checkpointing(self.encoder)
        enable_codesage_selective_checkpointing(
            self.encoder, 
            checkpoint_ratio=gradient_checkpointing_kwargs.get("checkpoint_ratio", 1.0) if gradient_checkpointing_kwargs else 1.0
        )

    def gradient_checkpointing_disable(self):
        disable_codesage_block_checkpointing(self.encoder)

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
        
        # # 3. Calculate Attention Scores -> (batch_size, max_chunks)
        # attn_scores = self.attention(bag_embeddings).squeeze(-1) 
        
        # # Mask out the padding chunks so they receive 0 attention
        # if chunk_mask is not None:
        #     attn_scores = attn_scores.masked_fill(chunk_mask == 0, -1e9)
            
        # attn_weights = torch.softmax(attn_scores, dim=1) 
        
        # # 4. Aggregate chunks into a single function-level embedding
        # # bmm: (batch, 1, max_chunks) x (batch, max_chunks, hidden_size) -> (batch, 1, hidden_size)
        # function_representation = torch.bmm(attn_weights.unsqueeze(1), bag_embeddings).squeeze(1)

        function_representation, attn_weights = self.attention(bag_embeddings, chunk_mask)
        
        # 5. Classify the entire function
        logits = self.classifier(function_representation)
        
        # Compute loss natively so Hugging Face Trainer can use it seamlessly
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.float().view_as(logits))

        out = {
            "loss": loss, 
            "logits": logits
        }

        if not self.training:
            out["attention_weights"] = function_representation
            
        return out