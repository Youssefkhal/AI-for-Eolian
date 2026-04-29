import json
import os
import random
from typing import Dict, Iterator

import torch
import torch.nn.functional as F

from config import Config
from data import simulate_dataset, split_dataset
from model import SlotTransformer


def _make_decoder_input(f0: torch.Tensor) -> torch.Tensor:
    while f0.dim() > 2:
        f0 = f0.squeeze(-1)
    zero = torch.zeros(f0.shape[0], 1, device=f0.device)
    shifted = torch.cat([zero, f0[:, :-1]], dim=1)
    return shifted.unsqueeze(-1)


def _iter_batches(data: Dict[str, torch.Tensor], batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
    n = data["x_static"].shape[0]
    idx = torch.randperm(n, device=data["x_static"].device)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield {k: v[batch_idx] for k, v in data.items()}


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((pred - target) ** 2).item()
    nrmse = torch.sqrt(torch.mean((pred - target) ** 2)) / (torch.mean(target.abs()) + 1e-8)
    mape = torch.mean((pred - target).abs() / (target.abs() + 1e-8))
    r2 = 1.0 - torch.sum((target - pred) ** 2) / (torch.sum((target - torch.mean(target)) ** 2) + 1e-8)
    return {
        "mse": float(mse),
        "nrmse": float(nrmse.item()),
        "mape": float(mape.item()),
        "r2": float(r2.item()),
    }


def train() -> Dict[str, object]:
    cfg = Config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fem = simulate_dataset(cfg.n_fem, cfg.seq_len, cfg.n_yield, noise=0.0, device=device)
    analytical = simulate_dataset(cfg.n_analytical, cfg.seq_len, cfg.n_yield, noise=0.02, device=device)
    experimental = simulate_dataset(cfg.n_experimental, cfg.seq_len, cfg.n_yield, noise=0.05, device=device)
    field = simulate_dataset(cfg.n_field, cfg.seq_len, cfg.n_yield, noise=0.08, device=device)
    holdout = simulate_dataset(cfg.n_holdout_scaled, cfg.seq_len, cfg.n_yield, noise=0.06, device=device)

    fem_train, fem_test = split_dataset(fem, cfg.train_split)
    analytical_train, analytical_test = split_dataset(analytical, cfg.train_split)
    experimental_train, _ = split_dataset(experimental, cfg.train_split)

    model = SlotTransformer(
        cfg.seq_len,
        cfg.n_yield,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_enc_layers,
        cfg.n_dec_layers,
        cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for epoch in range(cfg.train_epochs):
        for batch_a in _iter_batches(analytical_train, cfg.batch_size):
            batch_e = next(_iter_batches(experimental_train, cfg.batch_size))

            pred_a, aux_a, _ = model(batch_a["x_static"], batch_a["load_seq"], f0_prev=_make_decoder_input(batch_a["f0"]))
            pred_e, aux_e, _ = model(batch_e["x_static"], batch_e["load_seq"], f0_prev=_make_decoder_input(batch_e["f0"]))

            loss_data = 0.5 * F.mse_loss(pred_a, batch_a["f0"]) + 1.5 * F.mse_loss(pred_e, batch_e["f0"])
            loss_freq = 0.5 * F.mse_loss(aux_a["phys_f0"], batch_a["f0"]) + 1.5 * F.mse_loss(aux_e["phys_f0"], batch_e["f0"])

            dw = aux_a["w"][:, :, 1:] - aux_a["w"][:, :, :-1]
            loss_mon = F.relu(-dw).mean()
            loss_pos = F.relu(-aux_a["hij"]).mean()

            loss = loss_data + 0.5 * loss_freq + 0.2 * loss_mon + 0.2 * loss_pos

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_test, aux_test, attn = model(
            analytical_test["x_static"],
            analytical_test["load_seq"],
            f0_prev=_make_decoder_input(analytical_test["f0"]),
            return_attn=True,
        )
        metrics = _metrics(pred_test, analytical_test["f0"])

        holdout_pred, _, _ = model(
            holdout["x_static"],
            holdout["load_seq"],
            f0_prev=_make_decoder_input(holdout["f0"]),
        )
        holdout_metrics = _metrics(holdout_pred, holdout["f0"])

        field_pred, _, _ = model(
            field["x_static"],
            field["load_seq"],
            f0_prev=_make_decoder_input(field["f0"]),
        )
        field_metrics = _metrics(field_pred, field["f0"])

    heatmap = attn["slot_to_token"][0].detach().cpu().tolist()
    rollout = attn["dec_cross"][0].mean(dim=0).mean(dim=0).detach().cpu().tolist()

    payload = {
        "metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "field_metrics": field_metrics,
        "heatmap": heatmap,
        "rollout": rollout,
        "notes": "Synthetic output aligned to pages 29-37 requirements.",
    }

    out_path = os.path.join(os.path.dirname(__file__), cfg.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def main() -> None:
    result = train()
    print("done", result["metrics"])


if __name__ == "__main__":
    main()
