from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 16
    batch_size: int = 5
    lr: float = 1e-3
    steps: int = 500000
    dropout: float = 0.1
    vocab_size: int = 0
    average_loss_every: int = 2500
    num_heads: int = 8
    head_size: int = 4
    embd_dim: int = 407