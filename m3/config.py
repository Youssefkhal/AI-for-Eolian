from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 7
    seq_len: int = 50
    n_yield: int = 20
    d_model: int = 64
    n_heads: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    dropout: float = 0.1
    train_epochs: int = 8
    batch_size: int = 32
    lr: float = 5e-4
    train_split: float = 0.85

    n_fem: int = 200
    n_analytical: int = 200
    n_experimental: int = 30
    n_field: int = 10
    n_holdout_scaled: int = 3

    output_json: str = "outputs.json"
