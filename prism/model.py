import torch
from torch import nn
from prism.embedding import Embedding


class Model(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_size: int, 
        context_size: int, 
        num_heads: int = 12, 
        d_key: int = 64, 
        d_value: int = 64
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(context_size, embedding_size)
        self.masked_multi_head_attention = MultiHeadAttention(embedding_size, num_heads, d_key, d_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.embedding(x))


class PositionalEncoding(nn.Module):
    def __init__(self, context_size: int, embedding_size: int):
        super().__init__()
        self.positional_encoding = self._build_positional_encoding_matrix(context_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding + x

    def _build_positional_encoding_matrix(self, context_size, embedding_size):
        positions = torch.arange(context_size, dtype=torch.float32) 
        dimensions = torch.arange(embedding_size, dtype=torch.float32) 
        
        exponent = ((dimensions // 2) * 2) / embedding_size
        divisor = torch.pow(10000, exponent)
        angle_rates = positions.unsqueeze(1) / divisor
        
        # Initialize the positional encoding matrix and apply sine to even 
        # dimensions and cosine to odd dimensions.
        pos_encoding = torch.zeros_like(angle_rates)
        pos_encoding[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rates[:, 1::2])
        return pos_encoding


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        embedding_size: int, 
        num_heads: int, 
        d_key: int, 
        d_value: int,
        masked: bool = True
    ) -> None:
        super().__init__()
        self.masked = masked
        self.d_key = d_key
        self.q_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_key))
        self.k_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_key))
        self.v_proj = nn.Parameter(torch.randn(num_heads, embedding_size, d_value))
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Creates the query, key and value matrices for a given batch of 
        tokens.

        Args:
            x: A tensor of shape (batch_size, context_size, embedding_size)
            containing the token embeddings.

        Returns:
            A tensor of shape (batch_size, num_heads, context_size, d_value)
            containing the attention weights.
        """
        queries = torch.bmm(x, self.q_proj)
        keys = torch.bmm(x, self.k_proj)
        values = torch.bmm(x, self.v_proj)
        
        raw_attn_scores = torch.bmm(queries, torch.transpose(keys, -2, -1))
        scaled_attn_scores = raw_attn_scores / torch.sqrt(self.d_key)
        
        # mask the attention scores now
        if self.masked:
            mask = torch.triu(torch.ones_like(raw_attn_scores, dtype=torch.bool))
            scaled_attn_scores = scaled_attn_scores.masked_fill(mask, -float('inf'))

        # apply the softmax function
        attn_weights = torch.softmax(scaled_attn_scores, dim=-1)

        # apply the attention weights to the values
        out = torch.bmm(attn_weights, values)

        return out

# Add tests for all modules.