import math
from typing import Dict, Tuple

import torch


def _make_posdef_3x3(batch: int) -> torch.Tensor:
    a = torch.randn(batch, 3, 3)
    h = torch.matmul(a, a.transpose(-1, -2))
    h = h + 0.5 * torch.eye(3).unsqueeze(0)
    return h


def _make_nonneg_3x3(batch: int) -> torch.Tensor:
    a = torch.randn(batch, 3, 3)
    h = torch.matmul(a.abs(), a.abs().transpose(-1, -2))
    return h


def simulate_base(
    num_samples: int,
    seq_len: int,
    n_yield: int,
    device: torch.device,
    noise_scale: float = 0.0,
) -> Dict[str, torch.Tensor]:
    x_static = torch.randn(num_samples, 10, device=device)
    load_seq = torch.rand(num_samples, seq_len, 1, device=device)

    h0 = _make_posdef_3x3(num_samples).to(device)
    hn = _make_nonneg_3x3(num_samples * n_yield).to(device).view(num_samples, n_yield, 3, 3)

    a = torch.randn(num_samples, n_yield, 1, device=device) * 4.0
    b = torch.randn(num_samples, n_yield, 1, device=device)

    load_t = load_seq.transpose(1, 2)
    w = torch.sigmoid(a * load_t + b)

    h0_exp = h0.unsqueeze(1)
    hn_exp = hn.unsqueeze(2)
    w_exp = w.unsqueeze(-1).unsqueeze(-1)
    hij = h0_exp.unsqueeze(2) - torch.sum(w_exp * hn_exp, dim=1)
    hij = torch.clamp(hij, min=1e-3)

    m_eff = torch.exp(x_static[:, 0:1]) + 1.0
    k_eff = hij.diagonal(dim1=-2, dim2=-1).sum(-1)
    f0 = (1.0 / (2.0 * math.pi)) * torch.sqrt(k_eff / m_eff)

    if noise_scale > 0.0:
        x_static = x_static + noise_scale * torch.randn_like(x_static)
        load_seq = torch.clamp(load_seq + noise_scale * torch.randn_like(load_seq), 0.0, 1.0)

    return {
        "x_static": x_static,
        "load_seq": load_seq,
        "h0": h0,
        "hn": hn,
        "w": w,
        "f0": f0,
        "m_eff": m_eff,
    }


def simulate_fem_dataset(num_samples: int, seq_len: int, n_yield: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return simulate_base(num_samples, seq_len, n_yield, device, noise_scale=0.0)


def simulate_analytical_dataset(num_samples: int, seq_len: int, n_yield: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return simulate_base(num_samples, seq_len, n_yield, device, noise_scale=0.02)


def simulate_experimental_dataset(num_samples: int, seq_len: int, n_yield: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return simulate_base(num_samples, seq_len, n_yield, device, noise_scale=0.05)


def simulate_field_dataset(num_samples: int, seq_len: int, n_yield: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return simulate_base(num_samples, seq_len, n_yield, device, noise_scale=0.08)


def split_dataset(data: Dict[str, torch.Tensor], train_split: float) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    n = data["x_static"].shape[0]
    idx = torch.randperm(n)
    n_train = int(n * train_split)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    train = {k: v[train_idx] for k, v in data.items()}
    test = {k: v[test_idx] for k, v in data.items()}
    return train, test
