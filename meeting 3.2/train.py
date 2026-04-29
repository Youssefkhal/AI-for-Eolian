import os
import random
from typing import Dict, Iterator, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from config import TrainConfig
from data import (
    simulate_analytical_dataset,
    simulate_experimental_dataset,
    simulate_fem_dataset,
    simulate_field_dataset,
    split_dataset,
)
from losses import LossWeights, loss_terms, stiffness_pretrain_loss, total_loss
from metrics import bias_index, coefficient_of_variation, mape, nrmse, r2_score
from model import SelfCalibrationTransformer
from xai import attention_rollout, compute_attention_maps, save_heatmap, save_rollout


def _set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _make_decoder_input(f0: torch.Tensor) -> torch.Tensor:
    while f0.dim() > 2:
        f0 = f0.squeeze(-1)
    zero = torch.zeros(f0.shape[0], 1, device=f0.device)
    shifted = torch.cat([zero, f0[:, :-1]], dim=1)
    return shifted.unsqueeze(-1)


def _eval_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "nrmse": nrmse(pred, target),
        "mape": mape(pred, target),
        "r2": r2_score(pred, target),
        "bias": bias_index(pred, target),
    }


def _uncertainty_cov(model: SelfCalibrationTransformer, x_static: torch.Tensor, load_seq: torch.Tensor, f0_prev: torch.Tensor) -> torch.Tensor:
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(10):
            pred, _, _ = model(x_static, load_seq, f0_prev=f0_prev, return_attn=False)
            preds.append(pred)
    stacked = torch.stack(preds, dim=0)
    return coefficient_of_variation(stacked, dim=0).mean()


def _iter_batches(data: Dict[str, torch.Tensor], batch_size: int) -> Iterator[Dict[str, torch.Tensor]]:
    n = data["x_static"].shape[0]
    idx = torch.randperm(n, device=data["x_static"].device)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield {k: v[batch_idx] for k, v in data.items()}


def train(config: TrainConfig) -> None:
    print("[info] starting training pipeline...")
    if os.getenv("FAST_RUN", "0") == "1":
        config.n_fem = 60
        config.n_analytical = 60
        config.n_experimental = 12
        config.n_field = 4
        config.n_holdout_scaled = 2
        config.pretrain_epochs = 5
        config.train_epochs = 5
        config.d_model = 48
        config.n_enc_layers = 1
        config.n_dec_layers = 1
        print("[info] FAST_RUN enabled: using reduced sizes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        torch.set_num_interop_threads(1)
    print(f"[info] device: {device}")
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    fem = simulate_fem_dataset(config.n_fem, config.seq_len, config.n_yield, device)
    fem_train, fem_test = split_dataset(fem, config.train_split)
    print("[info] FEM dataset ready")

    analytical = simulate_analytical_dataset(config.n_analytical, config.seq_len, config.n_yield, device)
    analytical_train, analytical_test = split_dataset(analytical, config.train_split)
    print("[info] Analytical dataset ready")

    experimental = simulate_experimental_dataset(config.n_experimental, config.seq_len, config.n_yield, device)
    experimental_train, experimental_test = split_dataset(experimental, config.train_split)
    print("[info] Experimental dataset ready")

    holdout_scaled = simulate_experimental_dataset(config.n_holdout_scaled, config.seq_len, config.n_yield, device)
    field = simulate_field_dataset(config.n_field, config.seq_len, config.n_yield, device)
    print("[info] Holdout and field datasets ready")

    model = SelfCalibrationTransformer(
        config.seq_len,
        config.n_yield,
        config.d_model,
        config.n_heads,
        config.n_enc_layers,
        config.n_dec_layers,
        config.dropout,
    ).to(device)
    print("[info] model initialized")

    # Pretraining slot-attention on FEM (stiffness supervision)
    _set_trainable(model, False)
    _set_trainable(model.slot_attn, True)
    _set_trainable(model.h0_head, True)
    _set_trainable(model.hn_head, True)
    _set_trainable(model.gate_head, True)

    opt_pre = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr_pretrain)
    model.train()
    print("[info] pretraining slot-attention...")
    for epoch in range(config.pretrain_epochs):
        total_loss_val = 0.0
        for batch in _iter_batches(fem_train, config.batch_size):
            pred_f0, aux, _ = model(batch["x_static"], batch["load_seq"], f0_prev=_make_decoder_input(batch["f0"]))
            loss = stiffness_pretrain_loss(aux["h0"], batch["h0"], aux["hn"], batch["hn"])

            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()
            total_loss_val += loss.item()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                pred_eval, aux_eval, _ = model(fem_test["x_static"], fem_test["load_seq"], f0_prev=_make_decoder_input(fem_test["f0"]))
                loss_eval = stiffness_pretrain_loss(aux_eval["h0"], fem_test["h0"], aux_eval["hn"], fem_test["hn"])
            avg_loss = total_loss_val / max(1, fem_train["x_static"].shape[0] // config.batch_size)
            print(f"pretrain epoch={epoch+1} loss={avg_loss:.4f} test_loss={loss_eval.item():.4f}")
        if (epoch + 1) % 1 == 0:
            print(f"[running] pretrain epoch {epoch+1}/{config.pretrain_epochs}")

    # Full training (analytical + experimental) with physics losses
    _set_trainable(model, True)
    slot_params = list(model.slot_attn.parameters())
    slot_param_ids = {id(p) for p in slot_params}
    base_params = [p for p in model.parameters() if id(p) not in slot_param_ids]
    opt_train = torch.optim.Adam(
        [
            {"params": base_params, "lr": config.lr_train},
            {"params": slot_params, "lr": config.lr_train * config.slot_lr_scale},
        ]
    )
    weights = LossWeights()

    model.train()
    print("[info] full training (analytical + experimental)...")
    for epoch in range(config.train_epochs):
        total_loss_val = 0.0
        exp_batches = list(_iter_batches(experimental_train, config.batch_size))
        for i, batch_a in enumerate(_iter_batches(analytical_train, config.batch_size)):
            batch_e = exp_batches[i % len(exp_batches)]

            f0_prev_analytical = _make_decoder_input(batch_a["f0"])
            pred_a, aux_a, _ = model(batch_a["x_static"], batch_a["load_seq"], f0_prev=f0_prev_analytical)
            l_data_a, losses_a = loss_terms(pred_a, batch_a["f0"], aux_a)

            f0_prev_exp = _make_decoder_input(batch_e["f0"])
            pred_e, aux_e, _ = model(batch_e["x_static"], batch_e["load_seq"], f0_prev=f0_prev_exp)
            l_data_e, losses_e = loss_terms(pred_e, batch_e["f0"], aux_e)

            l_data = 0.5 * l_data_a + 1.5 * l_data_e
            losses = {
                "freq": 0.5 * losses_a["freq"] + 1.5 * losses_e["freq"],
                "mon": 0.5 * losses_a["mon"] + 1.5 * losses_e["mon"],
                "pos": 0.5 * losses_a["pos"] + 1.5 * losses_e["pos"],
            }
            l_total = total_loss(l_data, losses, weights)

            opt_train.zero_grad()
            l_total.backward()
            opt_train.step()
            total_loss_val += l_total.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_eval, aux_eval, _ = model(analytical_test["x_static"], analytical_test["load_seq"], f0_prev=_make_decoder_input(analytical_test["f0"]))
                metrics_eval = _eval_metrics(pred_eval, analytical_test["f0"])
            model.train()
            print(
                f"epoch={epoch+1} total={l_total.item():.4f} nrmse={metrics_eval['nrmse']:.4f} "
                f"mape={metrics_eval['mape']:.4f} r2={metrics_eval['r2']:.4f}"
            )
        if (epoch + 1) % 1 == 0:
            print(f"[running] train epoch {epoch+1}/{config.train_epochs}")

    model.eval()
    with torch.no_grad():
        pred_slot, aux_slot, _ = model(fem_test["x_static"], fem_test["load_seq"], f0_prev=_make_decoder_input(fem_test["f0"]))
        slot_metrics = _eval_metrics(aux_slot["h0"].mean(dim=(-2, -1)), fem_test["h0"].mean(dim=(-2, -1)))

        pred_full, aux_full, _ = model(analytical_test["x_static"], analytical_test["load_seq"], f0_prev=_make_decoder_input(analytical_test["f0"]))
        full_metrics = _eval_metrics(pred_full, analytical_test["f0"])

        holdout_pred, _, _ = model(holdout_scaled["x_static"], holdout_scaled["load_seq"], f0_prev=_make_decoder_input(holdout_scaled["f0"]))
        holdout_metrics = _eval_metrics(holdout_pred, holdout_scaled["f0"])

        field_pred, _, _ = model(field["x_static"], field["load_seq"], f0_prev=_make_decoder_input(field["f0"]))
        field_metrics = _eval_metrics(field_pred, field["f0"])

    cov_val = _uncertainty_cov(model, analytical_test["x_static"], analytical_test["load_seq"], _make_decoder_input(analytical_test["f0"]))
    print("[info] validation and uncertainty done")

    os.makedirs(config.save_dir, exist_ok=True)
    print("[info] computing attention maps...")
    f0_prev_xai = _make_decoder_input(analytical_test["f0"])
    attn_maps = compute_attention_maps(model, analytical_test["x_static"], analytical_test["load_seq"], f0_prev=f0_prev_xai)
    rollout = attention_rollout(attn_maps)

    torch.save(attn_maps, os.path.join(config.save_dir, "attention_maps.pt"))
    torch.save(rollout, os.path.join(config.save_dir, "attention_rollout.pt"))
    torch.save({
        "slot_metrics": slot_metrics,
        "full_metrics": full_metrics,
        "holdout_metrics": holdout_metrics,
        "field_metrics": field_metrics,
        "cov": cov_val,
    }, os.path.join(config.save_dir, "metrics.pt"))

    save_heatmap(attn_maps["slot_to_token"], os.path.join(config.save_dir, "slot_attention.png"))
    save_rollout(rollout, os.path.join(config.save_dir, "attention_rollout.png"))
    print(f"[done] outputs saved to {config.save_dir}")


if __name__ == "__main__":
    train(TrainConfig())
