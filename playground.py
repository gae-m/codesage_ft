from transformers import AutoModel, AutoConfig
import torch

encoder_name = "codesage/codesage-large-v2"

print("Loading the model...")
config = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
config.attention_dropout_prob = 0.0
config.residual_dropout_prob = 0.0
config.embedding_dropout_prob = 0.0

# Load base CodeSage without the classification head
base_encoder = AutoModel.from_pretrained(
    encoder_name,
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.float16,  # Loads the frozen weights directly in 16-bit
    # Loads the frozen weights directly in 16-bit
)

print(base_encoder)

print("Testing the model with a dummy input...")

dummy_input = torch.randint(0, 100, (1, 16), dtype=torch.int32)
dummy_attention = torch.ones_like(dummy_input, dtype=torch.int32)

outputs = base_encoder(input_ids=dummy_input, attention_mask=dummy_attention)
print(outputs.pooler_output.shape)  # Should be (1, hidden_size)


