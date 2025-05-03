import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

