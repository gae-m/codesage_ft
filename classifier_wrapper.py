import torch
import torch.nn as nn

from cs_checkpointing import disable_codesage_block_checkpointing, enable_codesage_selective_checkpointing


class ResidualTokenAttentionPooling(nn.Module):
    """
    Preserves CodeSage's native masked mean pooling while adding a zero-initialized,
    regularized token-attention residual to amplify subtle patch edits.
    """

    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        # `bias=False` because softmax is shift-invariant (a bias term would have zero grad forever)
        self.token_scorer = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.token_scorer.weight)
        self.attn_dropout = nn.Dropout(dropout_prob)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = token_embeddings.float()
        mask_f = attention_mask.float()

        # 1. Exact CodeSage masked mean pooling (never lost)
        mean_pool = (h * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)

        # 2. Learned token-attention residual (starts at `mean_pool` at step 0)
        scores = self.token_scorer(h).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores, dim=1)).unsqueeze(-1)
        attn_pool = (h * weights).sum(dim=1)

        return 0.5 * (mean_pool + attn_pool)


# class GatedAttention(nn.Module):
#     """
#     A standalone Gated Attention module for Multiple Instance Learning.
#     Computes attention scores using a gated mechanism (tanh * sigmoid) 
#     and aggregates the bag of chunks into a single representation.
#     """
#     def __init__(self, hidden_size, attention_dim, dropout_prob=0.1):
#         super().__init__()
#         # Feature extractor (V)
#         self.attention_v = nn.Linear(hidden_size, attention_dim)
#         # Gate mechanism (U)
#         self.attention_u = nn.Linear(hidden_size, attention_dim)
#         # Regularization
#         self.attention_dropout = nn.Dropout(dropout_prob)
#         # Final scorer
#         self.attention_weights = nn.Linear(attention_dim, 1)

#     def forward(self, bag_embeddings, chunk_mask=None):
#         # bag_embeddings shape: (batch_size, max_chunks, hidden_size)

#         # 3. Calculate Attention Scores -> (batch_size, max_chunks)
#         # attn_scores = self.attention(bag_embeddings).squeeze(-1) 
        
#         # # Mask out the padding chunks so they receive 0 attention
#         # if chunk_mask is not None:
#         #     attn_scores = attn_scores.masked_fill(chunk_mask == 0, -1e9)
            
#         # attn_weights = torch.softmax(attn_scores, dim=1) 
        
#         # # 4. Aggregate chunks into a single function-level embedding
#         # # bmm: (batch, 1, max_chunks) x (batch, max_chunks, hidden_size) -> (batch, 1, hidden_size)
#         # function_representation = torch.bmm(attn_weights.unsqueeze(1), bag_embeddings).squeeze(1)
        
#         # 1. Feature extraction and gating
#         v = torch.tanh(self.attention_v(bag_embeddings))
#         u = torch.sigmoid(self.attention_u(bag_embeddings))
        
#         # 2. Element-wise multiplication to apply the gate
#         gated_features = v * u 
#         gated_features = self.attention_dropout(gated_features)
        
#         # 3. Project down to 1 score per chunk
#         attn_scores = self.attention_weights(gated_features).squeeze(-1) # (batch_size, max_chunks)
        
#         # 4. Mask out padded chunks so they receive -inf score (0 attention)
#         if chunk_mask is not None:
#             attn_scores = attn_scores.masked_fill(chunk_mask == 0, -1e9)
            
#         # 5. Convert scores to probabilities
#         attn_weights = torch.softmax(attn_scores, dim=1) # (batch_size, max_chunks)
        
#         # 6. Aggregate chunks into a single function-level embedding
#         # bmm: (batch, 1, max_chunks) x (batch, max_chunks, hidden_size) -> (batch, 1, hidden_size)
#         function_representation = torch.bmm(attn_weights.unsqueeze(1), bag_embeddings).squeeze(1)
        
#         return function_representation, attn_weights


class GatedAttention(nn.Module):
    """Ilse et al. (2018) Gated Attention pooling across chunk embeddings."""

    def __init__(self, hidden_size: int = 2048, attention_dim: int = 256, dropout_prob: float = 0.1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(hidden_size, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_size, attention_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_weights = nn.Linear(attention_dim, 1)

    def forward(
        self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute in float32 for numerical stability across softmax
        h = chunk_embeddings.float()
        gated = self.dropout(self.attention_V(h) * self.attention_U(h))
        scores = self.attention_weights(gated).squeeze(-1)  # (B, C)

        if chunk_mask is not None:
            scores = scores.masked_fill(~chunk_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=1)  # (B, C)
        bag_representation = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return bag_representation, attn_weights


class AttentionMILCodeSage(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, attention_dim=256, mlp_dropout=0.1, pooling_dropout=0.1):
        super().__init__()
        self.encoder = encoder
        
        # The Attention Network: Learns to score the importance of each chunk
        # self.attention = nn.Sequential(
        #     nn.Linear(hidden_size, attention_dim),
        #     nn.Tanh(),
        #     nn.Linear(attention_dim, 1)
        # )
        self.pooling = ResidualTokenAttentionPooling(hidden_size=hidden_size, dropout_prob=pooling_dropout)
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
        # Masked mean pooling in float32 (matches CodeSage-v2 pretraining & avoids bfloat16/float32 mismatches)
        # token_embeddings = outputs.last_hidden_state.float()
        # # token embeddings shape: (batch_size * max_chunks, seq_len, hidden_size)
        # mask_expanded = flat_attention_mask.unsqueeze(-1).float()
        # chunk_embeddings = (
        #     (token_embeddings * mask_expanded).sum(dim=1)
        #     / mask_expanded.sum(dim=1).clamp(min=1.0)
        # )
        chunk_embeddings = self.pooling(outputs.last_hidden_state, flat_attention_mask)

        
        # 2. Reshape back into bags -> (batch_size, max_chunks, hidden_size)
        bag_embeddings = chunk_embeddings.view(batch_size, max_chunks, -1)

        # 3. Apply attention to get a single representation per function
        function_representation, attn_weights = self.attention(bag_embeddings, chunk_mask)
        
        # 4. Classify the entire function
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
            out["attention_weights"] = attn_weights
            
        return out


class PairedAttentionMILCodeSage(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        attention_dim: int,
        classifier_hidden_dim: int,
        lambda_rank_loss: float,
        lambda_attn_loss: float,
        return_attention: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "config"):
            self.encoder.config.use_cache = False

        self.lambda_rank_loss = lambda_rank_loss
        self.lambda_attn_loss = lambda_attn_loss
        self.return_attention = return_attention

        self.attention = GatedAttention(hidden_size, attention_dim, dropout_prob=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_dim, 1),
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.rank_loss = nn.SoftMarginLoss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
        diff_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size, max_chunks, seq_len = input_ids.size()

        # Eager mode optimization: trim trailing all-pad chunk columns to save encoder FLOPs
        # gm: this sohuld be useless if the collator is already doing this

        # if chunk_mask is not None:
        #     active_chunks = max(int(chunk_mask.sum(dim=1).max().item()), 1)
        #     input_ids = input_ids[:, :active_chunks, :]
        #     attention_mask = attention_mask[:, :active_chunks, :]
        #     chunk_mask = chunk_mask[:, :active_chunks]
        #     if diff_mask is not None:
        #         diff_mask = diff_mask[:, :active_chunks]
        #     max_chunks = active_chunks

        flat_ids = input_ids.reshape(-1, seq_len)
        flat_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)

        # Masked mean pooling in float32 (matches CodeSage-v2 pretraining & avoids bf16 mismatch)
        token_embeddings = outputs.last_hidden_state.float()
        mask_expanded = flat_mask.unsqueeze(-1).float()
        chunk_embeddings = (
            (token_embeddings * mask_expanded).sum(dim=1)
            / mask_expanded.sum(dim=1).clamp(min=1.0)
        ).view(batch_size, max_chunks, -1)

        function_representation, attn_weights = self.attention(chunk_embeddings, chunk_mask)
        logits = self.classifier(function_representation).squeeze(-1)

        loss = None
        if labels is not None:
            # 1. Standard pointwise BCE loss (keeps logits calibrated around threshold 0.0)
            loss = self.bce_loss(logits, labels.float())

            # Apply pairwise & diff losses only when paired batches are active in training
            if self.training and diff_mask is not None and (batch_size % 2 == 0):
                # 2. Pairwise logistic ranking loss: enforces logit(vuln) > logit(patch)
                pair_logits = logits.view(-1, 2)
                z_vuln, z_patch = pair_logits[:, 0], pair_logits[:, 1]
                rank_loss = F.softplus(-(z_vuln - z_patch)).mean()
                loss = loss + self.lambda_rank_loss * rank_loss

                # 3. Diff-guided attention loss: focuses attention onto modified chunks
                if chunk_mask is not None and self.lambda_attn_loss > 0.0:
                    num_diff = diff_mask.sum(dim=1)
                    num_valid = chunk_mask.float().sum(dim=1)
                    # Only supervise bags that have both modified and unmodified chunks
                    has_contrast = (num_diff > 0.0) & (num_diff < num_valid)
                    changed_attn_mass = (attn_weights.float() * diff_mask).sum(dim=1)
                    # `+ 1e-6` preserves non-zero gradients even when initial mass approaches 0
                    per_bag_attn_loss = -torch.log(changed_attn_mass + 1e-6)
                    attn_loss = torch.where(
                        has_contrast, per_bag_attn_loss, torch.zeros_like(per_bag_attn_loss)
                    ).mean()
                    loss = loss + self.attn_loss_weight * attn_loss

        output: dict[str, torch.Tensor | None] = {"loss": loss, "logits": logits}
        if not self.training and self.return_attention:
            output["attention_weights"] = attn_weights

        return output