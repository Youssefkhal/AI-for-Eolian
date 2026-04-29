import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    data: float = 1.0
    freq: float = 0.5
    mon: float = 0.2
    pos: float = 0.2


def loss_terms(
    pred_f0: torch.Tensor,
    true_f0: torch.Tensor,
    aux: dict,
) -> Tuple[torch.Tensor, dict]:
    l_data = F.mse_loss(pred_f0, true_f0)

    k_eff = aux["hij"].diagonal(dim1=-2, dim2=-1).sum(-1)
    f_analytical = (1.0 / (2.0 * math.pi)) * torch.sqrt(k_eff / aux["m_eff"])
    l_freq = F.mse_loss(f_analytical, true_f0)

    w = aux["w"]
    dw = w[:, :, 1:] - w[:, :, :-1]
    l_mon = F.relu(-dw).mean()

    l_pos = F.relu(-aux["hij"]).mean()

    return l_data, {"freq": l_freq, "mon": l_mon, "pos": l_pos}


def stiffness_pretrain_loss(pred_h0: torch.Tensor, true_h0: torch.Tensor, pred_hn: torch.Tensor, true_hn: torch.Tensor) -> torch.Tensor:
    l_h0 = F.mse_loss(pred_h0, true_h0)
    l_hn = F.mse_loss(pred_hn, true_hn)
    return l_h0 + l_hn


def total_loss(l_data: torch.Tensor, losses: Dict[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    return (
        weights.data * l_data
        + weights.freq * losses["freq"]
        + weights.mon * losses["mon"]
        + weights.pos * losses["pos"]
    )
