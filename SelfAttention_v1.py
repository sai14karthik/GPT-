import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # Compute Queries, Keys, and Values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compute attention scores
        attn_scores = queries @ keys.T

        # Normalize scores
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5),
            dim=-1
        )

        # Compute context vectors
        context_vec = attn_weights @ values

        return context_vec