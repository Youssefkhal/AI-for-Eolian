from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 7
    seq_len: int = 50
    n_yield: int = 20
    d_model: int = 64
    n_heads: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    dropout: float = 0.1

    n_fem: int = 300
    n_analytical: int = 300
    n_experimental: int = 30
    n_field: int = 10
    n_holdout_scaled: int = 3

    batch_size: int = 64

    pretrain_epochs: int = 20
    train_epochs: int = 30
    lr_pretrain: float = 1e-3
    lr_train: float = 5e-4
    slot_lr_scale: float = 0.2

    train_split: float = 0.85
    test_split: float = 0.15

    save_dir: str = "outputs"

    mape_slot_threshold: float = 0.10
    bias_slot_threshold: float = 0.05
    mape_f0_threshold: float = 0.01
    cov_threshold: float = 0.10
