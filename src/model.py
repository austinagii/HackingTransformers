from torch import nn

Query = Vector
Key = Vector
Value = Vector 
Output = Vector


class ScaledDotAttention(nn.Module):
    def __init__(self, queries: list[Query], values_by_key: dict[Key, Value]) -> None:

        
class PositionalEncoding(nn.Module):
    pass


class MultiHeadAttention(nn.Module):
    pass


class PositionalFeedForward(nn.Module):
    pass
