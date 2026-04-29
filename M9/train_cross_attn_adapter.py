"""
M9 + Decoder-Style Cross-Attention Adapter
==========================================

Goal:
  Keep the current M9 architecture intact, add one new cross-attention block
  inspired by the PDF's SCm->WTm encoder-decoder coupling, and train only that
  new block.

Interpretation of the PDF:
  - Current M9 slot stack acts as the support-condition encoder (SCm).
  - A new decoder-style cross-attention module queries that frozen support
    representation before prediction.
  - Only the new decoder/query cross-attention path is trainable.

Saved artifacts:
  - pile_model_cross_attn_adapter.pth
  - comparison_cross_attn_adapter.json
  - cross_attn_adapter_config.pkl
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from train import (
    DEVICE,
    M6_DIR,
    NUM_STEPS,
    SCRIPT_DIR,
    SlotAttentionDegradation,
    SlotAttentionPsiModel,
    compute_metrics,
    inverse_transform_values,
    load_and_group_data,
)


BASE_MODEL_PATH = os.path.join(SCRIPT_DIR, "pile_model.pth")
OUT_MODEL_PATH = os.path.join(SCRIPT_DIR, "pile_model_cross_attn_adapter.pth")
OUT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "cross_attn_adapter_config.pkl")
OUT_COMPARISON_PATH = os.path.join(SCRIPT_DIR, "comparison_cross_attn_adapter.json")


class DecoderCrossAttentionAdapter(nn.Module):
    """Decoder-like cross-attention that queries frozen encoder slots.

    - Query tokens are learned decoder queries.
    - Key/value come from the frozen M9 slot representation.
    - Output is fused back into the base slot sequence with a learnable gate.
    """

    def __init__(self, num_slots: int, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.decoder_queries = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, encoder_slots: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_slots.size(0)
        queries = self.decoder_queries.expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(queries, encoder_slots, encoder_slots)
        decoded = self.cross_norm(queries + attn_out)
        gate = torch.sigmoid(self.fuse_gate)
        return self.fuse_norm(encoder_slots + gate * decoded)


class FrozenM9WithCrossAttention(nn.Module):
    """Frozen base M9 + trainable decoder-style cross-attention adapter."""

    def __init__(self, base_model: SlotAttentionPsiModel, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.adapter = DecoderCrossAttentionAdapter(
            num_slots=self.base.max_seq_len,
            d_model=self.base.d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        for param in self.base.parameters():
            param.requires_grad = False

    def encode_slots(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_embed = self.base.input_embed(x).unsqueeze(1)
        initial = self.base.initial_slot.expand(batch_size, -1, -1)
        drops = self.base.reconstruct_drop_slots(batch_size)
        slots = torch.cat([initial, drops], dim=1)

        for _ in range(self.base.num_iterations):
            cross_out, _ = self.base.cross_attn(slots, x_embed, x_embed)
            slots = self.base.cross_norm(slots + cross_out)
            self_out, _ = self.base.self_attn(slots, slots, slots)
            slots = self.base.self_norm(slots + self_out)
            slots = self.base.mlp_norm(slots + self.base.slot_mlp(slots))

        return slots

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = self.base.max_seq_len

        slots = self.encode_slots(x)
        slots = self.adapter(slots)

        init_pred = self.base.initial_proj(slots[:, 0:1, :])
        raw_drops = self.base.drop_proj(slots[:, 1:, :])

        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = torch.abs(raw_drops[:, :, 2:3])
        constrained = torch.cat([drops_kl_kr, drops_klr], dim=2)

        return torch.cat([init_pred, init_pred + torch.cumsum(constrained, dim=1)], dim=1)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def evaluate_model(model, x_test, y_test_orig, scaler_y, label: str):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(x_test), seq_len=NUM_STEPS).cpu().numpy()
    pred_orig = inverse_transform_values(pred_scaled, scaler_y)

    var_names = ["KL", "KR", "KLR"]
    overall = compute_metrics(y_test_orig, pred_orig)
    per_variable = {
        name: compute_metrics(y_test_orig[:, :, idx], pred_orig[:, :, idx])
        for idx, name in enumerate(var_names)
    }

    print(f"\n  {label}:")
    print(
        f"    Overall: R²={overall['r2']:.4f}  RMSE={overall['rmse']:.4e}  MAE={overall['mae']:.4e}"
    )
    for name in var_names:
        metric = per_variable[name]
        print(
            f"    {name:>3}: R²={metric['r2']:.4f}  RMSE={metric['rmse']:.4e}  MAE={metric['mae']:.4e}"
        )

    return {"overall": overall, "per_variable": per_variable}, pred_orig


def train_adapter(
    model: FrozenM9WithCrossAttention,
    teacher: SlotAttentionDegradation,
    x_train,
    x_val,
    y_train,
    y_val,
    epochs: int,
    batch_size: int,
    lr: float,
):
    print(f"\n{'=' * 60}")
    print("STAGE D: Decoder-Style Cross-Attention Adapter")
    print(f"{'=' * 60}")
    print("  Base M9 parameters: frozen")
    print(f"  Trainable adapter params: {model.trainable_parameter_count():,}")
    print("  Architecture: learned decoder queries -> cross-attend frozen M9 slots")
    print(f"  Device: {DEVICE}")

    teacher.to(DEVICE)
    teacher.eval()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=0.001,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=40, factor=0.5, min_lr=1e-6
    )

    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    x_val_t = torch.FloatTensor(x_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)

    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()
    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mono = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = teacher(x_batch, seq_len=NUM_STEPS)

            pred = model(x_batch, seq_len=NUM_STEPS)

            loss_distill = mse(pred, teacher_out)
            loss_seq = huber(pred, y_batch)
            loss_initial = mse(pred[:, 0, :], y_batch[:, 0, :]) * 5.0
            diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
            diff_target = y_batch[:, 1:, :] - y_batch[:, :-1, :]
            loss_shape = huber(diff_pred, diff_target)

            diff_kl_kr = pred[:, 1:, :2] - pred[:, :-1, :2]
            diff_klr = pred[:, 1:, 2:3] - pred[:, :-1, 2:3]
            loss_mono = torch.relu(diff_kl_kr).mean() + torch.relu(-diff_klr).mean()

            loss = loss_distill + loss_seq + loss_initial + loss_shape + 0.2 * loss_mono
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 2.0
            )
            optimizer.step()

            total_loss += loss.item()
            total_mono += loss_mono.item()

        total_loss /= len(train_loader)
        total_mono /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t, seq_len=NUM_STEPS)
            val_loss = mse(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 50 == 0:
            gate_value = torch.sigmoid(model.adapter.fuse_gate.detach()).item()
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:>4}/{epochs}: Loss={total_loss:.6f}  "
                f"Val={val_loss:.6f}  Mono={total_mono:.4f}  Gate={gate_value:.4f}  LR={lr_now:.2e}"
            )

        if patience_count >= 120:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.to("cpu")
    teacher.to("cpu")
    print(f"  Adapter training complete. Best val loss: {best_val_loss:.6f}")
    return model


def build_datasets():
    excel_path = os.path.join(os.path.dirname(SCRIPT_DIR), "M8", "REAL DATA.xlsx")
    if not os.path.exists(excel_path):
        excel_path = os.path.join(SCRIPT_DIR, "REAL DATA.xlsx")
    if not os.path.exists(excel_path):
        excel_path = os.path.join(M6_DIR, "REAL DATA.xlsx")

    x_list, y_list, input_cols, output_cols = load_and_group_data(excel_path)
    x_array = np.array(x_list)
    y_array = np.array(y_list)

    scaler_x = RobustScaler()
    x_scaled = scaler_x.fit_transform(x_array)

    y_sign = np.sign(y_array)
    y_log = y_sign * np.log1p(np.abs(y_array))
    scaler_y = RobustScaler()
    y_flat = y_log.reshape(-1, 3)
    y_scaled = scaler_y.fit_transform(y_flat).reshape(y_log.shape)

    indices = np.arange(len(x_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    return {
        "x_array": x_array,
        "y_array": y_array,
        "x_scaled": x_scaled,
        "y_scaled": y_scaled,
        "x_train": x_scaled[train_idx],
        "x_test": x_scaled[test_idx],
        "y_train": y_scaled[train_idx],
        "y_test": y_scaled[test_idx],
        "y_test_orig": y_array[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "input_cols": input_cols,
        "output_cols": output_cols,
    }


def load_teacher(input_size: int):
    teacher = SlotAttentionDegradation(
        input_size=input_size,
        d_model=64,
        num_heads=4,
        num_slots=NUM_STEPS,
        max_seq_len=NUM_STEPS,
        dropout=0.1,
        num_iterations=3,
    )
    teacher.load_state_dict(
        torch.load(os.path.join(M6_DIR, "pile_model.pth"), map_location="cpu", weights_only=True)
    )
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def load_base_m9(input_size: int):
    psi_config = joblib.load(os.path.join(SCRIPT_DIR, "psi_config.pkl"))
    base_model = SlotAttentionPsiModel(
        input_size=input_size,
        d_model=64,
        num_heads=4,
        max_seq_len=NUM_STEPS,
        dropout=0.1,
        num_iterations=3,
        num_prototypes=psi_config["num_prototypes"],
        relation_matrix=np.array(psi_config["relation_matrix"]),
    )
    base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location="cpu", weights_only=True))
    return base_model, psi_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1

    print("=" * 60)
    print("M9 + Decoder-Style Cross-Attention Adapter")
    print("=" * 60)
    print("  Base M9 architecture: kept intact")
    print("  New module: decoder-style cross-attention adapter")
    print("  Trainable scope: only new adapter block")

    data = build_datasets()
    print(f"\n  Train: {len(data['x_train'])}, Test: {len(data['x_test'])}, Steps: {NUM_STEPS}")

    teacher = load_teacher(data["x_train"].shape[1])
    base_model, psi_config = load_base_m9(data["x_train"].shape[1])
    model = FrozenM9WithCrossAttention(base_model)

    base_metrics, _ = evaluate_model(base_model, data["x_test"], data["y_test_orig"], data["scaler_y"], "Base M9")

    model = train_adapter(
        model,
        teacher,
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    adapter_metrics, _ = evaluate_model(
        model, data["x_test"], data["y_test_orig"], data["scaler_y"], "M9 + Cross-Attention Adapter"
    )

    torch.save(model.state_dict(), OUT_MODEL_PATH)
    joblib.dump(
        {
            "base_model_path": BASE_MODEL_PATH,
            "num_slots": NUM_STEPS,
            "d_model": 64,
            "num_heads": 4,
            "relation_matrix": psi_config["relation_matrix"],
            "num_prototypes": psi_config["num_prototypes"],
        },
        OUT_CONFIG_PATH,
    )

    comparison = {
        "base_m9": base_metrics,
        "m9_cross_attn_adapter": {
            **adapter_metrics,
            "params_total": int(sum(parameter.numel() for parameter in model.parameters())),
            "params_trainable": int(model.trainable_parameter_count()),
            "train_scope": "adapter_only",
        },
    }
    with open(OUT_COMPARISON_PATH, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    print(f"\n{'=' * 60}")
    print("Saved artifacts")
    print(f"  {OUT_MODEL_PATH}")
    print(f"  {OUT_CONFIG_PATH}")
    print(f"  {OUT_COMPARISON_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()