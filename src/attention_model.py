import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

from src.constants import ALPHABET_SIZE


class AttentionFFNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, input_seq):
        attention_outputs, _ = self.attention(input_seq, input_seq, input_seq)
        res_plus_attention_outputs = self.norm1(attention_outputs + input_seq)
        ffn_outputs = self.linear2(self.relu(self.linear1(res_plus_attention_outputs)))
        ffn_with_overall = ffn_outputs + ffn_outputs.mean(-2, keepdims=True)
        outputs_with_residual = self.norm2(ffn_with_overall + res_plus_attention_outputs)
        return outputs_with_residual


class AttentionModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads):
        super().__init__()

        self.embedding = nn.Linear(20, hidden_dim)
        self.attention_ffn_layers = nn.Sequential(
                # AttentionFFNLayer(hidden_dim),
                # AttentionFFNLayer(hidden_dim),
                # AttentionFFNLayer(hidden_dim),
                *[TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)]
            )
        self.linear_head = nn.Linear(hidden_dim, ALPHABET_SIZE)

    def forward(self, text_inds):
        symbol_locations = (torch.arange(ALPHABET_SIZE, dtype=torch.int8).unsqueeze(1) == text_inds.unsqueeze(-2)).float()
        embeddings = self.embedding(symbol_locations)
        embeddings_with_overall = embeddings + embeddings.sum(-2, keepdims=True)
        attention_ffn_layers_outputs = self.attention_ffn_layers(embeddings_with_overall)
        return self.linear_head(attention_ffn_layers_outputs)
