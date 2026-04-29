"""
M9: SwiGLU Ψ-NN Applied to Pile Stiffness Degradation
=======================================================
Full 3-stage pipeline with SwiGLU upgrade:
  Stage A — Train student via distillation (SwiGLU MLP)
  Stage B — Structure discovery (clustering student slots → k*, R, centroids)
  Stage C — Train SwiGLU Ψ-Model from discovered structure

SwiGLU = Swish-Gated Linear Unit — gate × value → output.
Used in LLaMA/PaLM. Same parameter budget as M8 (hidden=32 ≈ M8's 48),
but provably more expressive due to element-wise gating.

Teacher: M6 SlotAttentionDegradation (frozen, 56,646 params)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M6_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), 'M6')

NUM_STEPS   = 21
STEP_INDICES = np.round(np.linspace(0, 43, NUM_STEPS)).astype(int)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SwiGLU hidden dimension — keeps param budget ≈ M8's bottleneck-48 MLP
# M8 EfficientSlotMLP(48): 64×48+48 + 48×64+64 = 6,256 params
# M9 SwiGLUSlotMLP(32)  :  64×32+32 + 64×32+32 + 32×64+64 = 6,272 params  ✓
SWIGLU_DIM = 32


# ─────────────────────────────────────────────────────
# SwiGLU Slot MLP  (M9 core upgrade)
# ─────────────────────────────────────────────────────

class SwiGLUSlotMLP(nn.Module):
    """
    SwiGLU feed-forward block:
        out = W_out( SiLU(W_gate(x)) ⊙ W_val(x) )

    Two parallel linear projections (gate + value), element-wise product,
    then project back.  Gating lets the network learn WHICH features to
    propagate — outperforms plain GELU MLP at the same parameter budget.

    Param count with d=64, hidden=32:
        W_gate : 64×32 + 32  = 2 080
        W_val  : 64×32 + 32  = 2 080
        W_out  : 32×64 + 64  = 2 112
        Total  :               6 272   (cf. M8 EfficientSlotMLP: 6 256)
    """

    def __init__(self, d_model: int = 64, hidden: int = SWIGLU_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, hidden)
        self.W_val  = nn.Linear(d_model, hidden)
        self.W_out  = nn.Linear(hidden,  d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x))      # Swish gate
        val  = self.W_val(x)               # value branch
        return self.W_out(self.drop(gate * val))


# ─────────────────────────────────────────────────────
# M6 Teacher (frozen — identical copy for weight load)
# ─────────────────────────────────────────────────────

class SlotAttentionDegradation(nn.Module):
    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21,
                 max_seq_len=21, dropout=0.1, num_iterations=3):
        super().__init__()
        self.num_slots     = num_slots
        self.d_model       = d_model
        self.max_seq_len   = max_seq_len
        self.num_iterations= num_iterations

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots      = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm  = nn.LayerNorm(d_model)
        self.slot_mlp   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model))
        self.mlp_norm   = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj    = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots   = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _  = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        initial    = self.initial_proj(slots[:, 0:1, :])
        raw_drops  = self.drop_proj(slots[:, 1:, :])
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr   =  torch.abs(raw_drops[:, :, 2:3])
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        return torch.cat([initial, initial + torch.cumsum(drops, dim=1)], dim=1)

    def get_refined_slots(self, x):
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots   = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _  = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        return slots


# ─────────────────────────────────────────────────────
# Stage A: Student Model (SwiGLU MLP + L1)
# ─────────────────────────────────────────────────────

class SlotAttentionStudent(nn.Module):
    """Student that imitates the teacher, with SwiGLU MLP and L1 slot reg."""

    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21,
                 max_seq_len=21, dropout=0.1, num_iterations=3):
        super().__init__()
        self.num_slots      = num_slots
        self.d_model        = d_model
        self.max_seq_len    = max_seq_len
        self.num_iterations = num_iterations

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots      = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm  = nn.LayerNorm(d_model)
        self.slot_mlp   = SwiGLUSlotMLP(d_model=d_model, hidden=SWIGLU_DIM, dropout=dropout)
        self.mlp_norm   = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj    = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots   = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _  = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        initial    = self.initial_proj(slots[:, 0:1, :])
        raw_drops  = self.drop_proj(slots[:, 1:, :])
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr   =  torch.abs(raw_drops[:, :, 2:3])
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        return torch.cat([initial, initial + torch.cumsum(drops, dim=1)], dim=1)

    def get_refined_slots(self, x):
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots   = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _  = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        return slots

    def l1_regularization(self):
        return torch.mean(torch.abs(self.slots))


# ─────────────────────────────────────────────────────
# Stage C: SwiGLU Ψ-Model  (M9 upgrade)
# ─────────────────────────────────────────────────────

class SlotAttentionPsiModel(nn.Module):
    """
    Ψ-NN structured model — M9 version:
      · SwiGLUSlotMLP  (d=64, hidden=32) replaces EfficientSlotMLP(48)
      · Learnable relation logits (same as M8): row-softmax → valid prob.
      · Physics-monotonic penalty applied in training loop (same as M8).
    """

    def __init__(self, input_size=8, d_model=64, num_heads=4,
                 max_seq_len=21, dropout=0.1, num_iterations=3,
                 num_prototypes=4, relation_matrix=None):
        super().__init__()
        self.d_model        = d_model
        self.max_seq_len    = max_seq_len
        self.num_iterations = num_iterations
        self.num_prototypes = num_prototypes
        self.num_drop_slots = max_seq_len - 1   # 20

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())

        self.initial_slot    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.prototype_slots = nn.Parameter(torch.randn(1, num_prototypes, d_model) * 0.02)

        # Learnable relation logits (identical to M8)
        if relation_matrix is not None:
            R_init = torch.FloatTensor(relation_matrix)
        else:
            R_init = torch.zeros(self.num_drop_slots, num_prototypes)
            spp = self.num_drop_slots // num_prototypes
            for p in range(num_prototypes):
                s = p * spp
                e = s + spp if p < num_prototypes - 1 else self.num_drop_slots
                R_init[s:e, p] = 1.0
        self.relation_logits = nn.Parameter(torch.log(R_init.clamp(min=1e-6)))
        self.slot_scales     = nn.Parameter(torch.ones(self.num_drop_slots, 1))

        # Shared attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm  = nn.LayerNorm(d_model)

        # ── M9 upgrade: SwiGLU instead of EfficientSlotMLP ──
        self.slot_mlp = SwiGLUSlotMLP(d_model=d_model, hidden=SWIGLU_DIM, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(d_model)

        # Prediction heads
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj    = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def get_relation_matrix(self):
        return torch.softmax(self.relation_logits, dim=1)

    def reconstruct_drop_slots(self, B):
        protos     = self.prototype_slots.expand(B, -1, -1)
        R          = self.get_relation_matrix()
        drop_slots = torch.matmul(R, protos)
        return drop_slots * self.slot_scales.unsqueeze(0)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)

        initial = self.initial_slot.expand(B, -1, -1)
        drops   = self.reconstruct_drop_slots(B)
        slots   = torch.cat([initial, drops], dim=1)

        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _  = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))

        init_pred  = self.initial_proj(slots[:, 0:1, :])
        raw_drops  = self.drop_proj(slots[:, 1:, :])

        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr   =  torch.abs(raw_drops[:, :, 2:3])
        constrained = torch.cat([drops_kl_kr, drops_klr], dim=2)

        return torch.cat([init_pred, init_pred + torch.cumsum(constrained, dim=1)], dim=1)


# ─────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────

def load_and_group_data(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    col_map = {'Dp/Lp': 'Dp_Lp', 'KL': 'kl', 'KR': 'kr', 'KLR': 'klr'}
    df.rename(columns=col_map, inplace=True)
    input_cols  = ['PI', 'Gmax', 'v', 'Dp', 'Tp', 'Lp', 'Ip', 'Dp_Lp']
    output_cols = ['kl', 'kr', 'klr']
    groups = df.groupby(input_cols, sort=False)
    X_list, Y_list = [], []
    for name, group in groups:
        outputs = group[output_cols].values
        if len(outputs) != 44:
            continue
        initial = outputs[0]
        drops   = outputs[1:]
        actual  = initial - np.cumsum(drops, axis=0)
        full_seq = np.vstack([initial, actual])
        subsampled = full_seq[STEP_INDICES]
        X_list.append(list(name))
        Y_list.append(subsampled)
    print(f"Loaded {len(X_list)} scenarios, {NUM_STEPS} steps each")
    return X_list, Y_list, input_cols, output_cols


def inverse_transform_values(scaled, scaler):
    flat   = scaled.reshape(-1, 3)
    log_v  = scaler.inverse_transform(flat)
    orig   = np.sign(log_v) * np.expm1(np.abs(log_v))
    return orig.reshape(scaled.shape)


def compute_metrics(y_true, y_pred):
    ft, fp = y_true.flatten(), y_pred.flatten()
    if len(ft) < 2:
        return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
    return {
        'r2':   float(r2_score(ft, fp)),
        'rmse': float(np.sqrt(mean_squared_error(ft, fp))),
        'mae':  float(mean_absolute_error(ft, fp)),
    }


# ─────────────────────────────────────────────────────
# Stage A: Train Student via Distillation
# ─────────────────────────────────────────────────────

def stage_a_distillation(teacher, X_train, X_val, Y_train, Y_val,
                         epochs=2000, batch_size=8, lr=0.001, mu=0.01):
    print(f"\n{'='*60}")
    print("STAGE A: Physics-Informed Distillation (SwiGLU Student)")
    print(f"{'='*60}")
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  L1 regularization mu = {mu}")
    print(f"  SwiGLU hidden = {SWIGLU_DIM}")
    print(f"  Device: {DEVICE}")

    teacher.eval()

    student = SlotAttentionStudent(
        input_size=X_train.shape[1], d_model=64, num_heads=4,
        num_slots=NUM_STEPS, max_seq_len=NUM_STEPS, dropout=0.1, num_iterations=3)
    student.to(DEVICE)

    n_student = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {n_student:,}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=60, factor=0.5, min_lr=1e-6)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    Y_val_t = torch.FloatTensor(Y_val).to(DEVICE)

    mse = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_out = teacher(X_batch, seq_len=NUM_STEPS)
            student_out = student(X_batch, seq_len=NUM_STEPS)

            loss_distill = mse(student_out, teacher_out)
            loss_data = mse(student_out, Y_batch)
            loss_l1 = student.l1_regularization()

            loss = loss_distill + 0.5 * loss_data + mu * loss_l1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)

        student.eval()
        with torch.no_grad():
            val_pred = student(X_val_t, seq_len=NUM_STEPS)
            val_teacher = teacher(X_val_t, seq_len=NUM_STEPS)
            val_loss = mse(val_pred, val_teacher).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 100 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            l1_val = student.l1_regularization().item()
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss:.6f}, "
                  f"ValDistill={val_loss:.6f}, L1={l1_val:.4f}, LR={lr_now:.2e}")

        if patience_cnt >= 150:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    student.load_state_dict(best_state)
    student.to('cpu')
    print(f"  Student trained. Best distillation loss: {best_val_loss:.6f}")
    return student


# ─────────────────────────────────────────────────────
# Stage B: Structure Discovery
# ─────────────────────────────────────────────────────

def stage_b_structure_discovery(student, teacher, X_data):
    print(f"\n{'='*60}")
    print("STAGE B: Structure Extraction & Clustering")
    print(f"{'='*60}")

    student.eval()
    teacher.eval()
    student.to('cpu')
    teacher.to('cpu')

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_data)
        student_slots = student.get_refined_slots(X_tensor)
        teacher_slots = teacher.get_refined_slots(X_tensor)

    drop_slots_student = student_slots[:, 1:, :].numpy()
    avg_drop_slots = drop_slots_student.mean(axis=0)  # [20, 64]
    print(f"\n  Drop slot vectors shape: {avg_drop_slots.shape}")

    slot_norms = np.linalg.norm(avg_drop_slots, axis=1)
    print(f"  Slot norms (min={slot_norms.min():.3f}, max={slot_norms.max():.3f}, "
          f"mean={slot_norms.mean():.3f})")

    norm_slots = avg_drop_slots / (np.linalg.norm(avg_drop_slots, axis=1, keepdims=True) + 1e-8)
    cosine_sim = norm_slots @ norm_slots.T
    print(f"  Cosine similarity: mean={cosine_sim.mean():.3f}, "
          f"avg off-diagonal={(cosine_sim.sum() - np.trace(cosine_sim)) / (20*19):.3f}")

    k_range = range(2, min(11, len(avg_drop_slots)))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(avg_drop_slots)
        inertias.append(km.inertia_)
        sil = silhouette_score(avg_drop_slots, labels)
        silhouettes.append(sil)

    best_k_idx = np.argmax(silhouettes)
    k_star = list(k_range)[best_k_idx]
    best_silhouette = silhouettes[best_k_idx]

    print(f"\n  Clustering results:")
    for i, k in enumerate(k_range):
        marker = " << BEST" if k == k_star else ""
        print(f"    k={k}: inertia={inertias[i]:.2f}, silhouette={silhouettes[i]:.3f}{marker}")
    print(f"\n  Optimal k* = {k_star} (silhouette = {best_silhouette:.3f})")

    km_final = KMeans(n_clusters=k_star, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(avg_drop_slots)
    centroids = km_final.cluster_centers_

    distances = np.zeros((20, k_star))
    for i in range(20):
        for j in range(k_star):
            distances[i, j] = np.linalg.norm(avg_drop_slots[i] - centroids[j])

    inv_dist = 1.0 / (distances + 1e-8)
    R = inv_dist / inv_dist.sum(axis=1, keepdims=True)

    for i in range(20):
        if R[i].max() > 0.8:
            dominant = R[i].argmax()
            R[i] = 0.0
            R[i, dominant] = 1.0

    cluster_info = {}
    for p in range(k_star):
        members = np.where(cluster_labels == p)[0]
        member_norms = slot_norms[members]
        cluster_info[f"Prototype_{p}"] = {
            'members': (members + 2).tolist(),
            'count': len(members),
            'avg_norm': float(member_norms.mean()),
            'centroid_norm': float(np.linalg.norm(centroids[p])),
        }
        print(f"\n  Prototype {p}: slots {(members+2).tolist()}")
        print(f"    Count: {len(members)}, Avg norm: {member_norms.mean():.3f}")

    discovery = {
        'k_star': k_star,
        'best_silhouette': float(best_silhouette),
        'cluster_labels': cluster_labels.tolist(),
        'slot_norms': slot_norms.tolist(),
        'cosine_similarity': cosine_sim.tolist(),
        'relation_matrix_init': R.tolist(),
        'centroids': centroids.tolist(),
        'cluster_info': cluster_info,
        'inertias': {str(k): float(v) for k, v in zip(k_range, inertias)},
        'silhouettes': {str(k): float(v) for k, v in zip(k_range, silhouettes)},
    }

    return k_star, R, centroids, discovery


# ─────────────────────────────────────────────────────
# Stage C: SwiGLU Ψ-Model Training
# ─────────────────────────────────────────────────────

def stage_c_swiglu_training(teacher, k_star, relation_matrix, centroids,
                             X_train, X_val, Y_train, Y_val,
                             epochs=2500, batch_size=8, lr=0.001):
    """
    Train the SwiGLU Ψ-model.
    Loss = L_distill + L_seq + 5·L_initial + L_shape
         + 0.02·L_rel_entropy  + 0.2·L_physics_mono
    Identical multi-objective to M8, only the slot MLP is SwiGLU.
    """
    print(f"\n{'='*60}")
    print("STAGE C: SwiGLU Ψ-Model Training (M9)")
    print(f"{'='*60}")

    psi_model = SlotAttentionPsiModel(
        input_size=X_train.shape[1], d_model=64, num_heads=4,
        max_seq_len=NUM_STEPS, dropout=0.1, num_iterations=3,
        num_prototypes=k_star, relation_matrix=relation_matrix)

    # Initialise prototype slots from discovered centroids
    with torch.no_grad():
        psi_model.prototype_slots.copy_(
            torch.FloatTensor(centroids).unsqueeze(0))

    psi_model.to(DEVICE)
    teacher.to(DEVICE)

    n_psi     = sum(p.numel() for p in psi_model.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Ψ-Model params : {n_psi:,}")
    print(f"  Teacher params : {n_teacher:,}")
    print(f"  Compression    : {(1 - n_psi / n_teacher) * 100:.1f}% fewer than M6")
    print(f"  SwiGLU hidden  : {SWIGLU_DIM}  (gate+val: 2×Linear(64,{SWIGLU_DIM}))")
    print(f"  Prototypes     : {k_star}")
    print(f"  Device         : {DEVICE}")

    teacher.eval()

    optimizer = torch.optim.AdamW(psi_model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=80, factor=0.5, min_lr=1e-6)

    train_ds     = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    Y_val_t = torch.FloatTensor(Y_val).to(DEVICE)

    mse   = nn.MSELoss()
    huber = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    best_state    = None
    patience_cnt  = 0

    for epoch in range(epochs):
        psi_model.train()
        total_loss = epoch_mono = epoch_rent = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = teacher(X_batch, seq_len=NUM_STEPS)

            psi_out = psi_model(X_batch, seq_len=NUM_STEPS)

            # Base multi-objective losses (identical to M8)
            loss_distill  = mse(psi_out, teacher_out)
            loss_seq      = huber(psi_out, Y_batch)
            loss_initial  = mse(psi_out[:, 0, :], Y_batch[:, 0, :]) * 5.0
            diff_pred     = psi_out[:, 1:, :]   - psi_out[:, :-1, :]
            diff_target   = Y_batch[:, 1:, :]   - Y_batch[:, :-1, :]
            loss_shape    = huber(diff_pred, diff_target)

            # Relation-entropy regularisation (sharpens prototype assignments)
            R_w = psi_model.get_relation_matrix()
            loss_rel_ent = -(R_w * torch.log(R_w + 1e-8)).sum(dim=1).mean()

            # Physics-monotonic penalty (KL↓, KR↓, KLR↑)
            diff_kl_kr = psi_out[:, 1:, :2]  - psi_out[:, :-1, :2]
            diff_klr   = psi_out[:, 1:, 2:3] - psi_out[:, :-1, 2:3]
            loss_mono  = torch.relu(diff_kl_kr).mean() + torch.relu(-diff_klr).mean()

            loss = (loss_distill + loss_seq + loss_initial + loss_shape
                    + 0.02 * loss_rel_ent + 0.2 * loss_mono)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(psi_model.parameters(), 2.0)
            optimizer.step()

            total_loss += loss.item()
            epoch_mono += loss_mono.item()
            epoch_rent += loss_rel_ent.item()

        n_b = len(train_loader)
        total_loss /= n_b;  epoch_mono /= n_b;  epoch_rent /= n_b

        psi_model.eval()
        with torch.no_grad():
            val_pred = psi_model(X_val_t, seq_len=NUM_STEPS)
            val_loss = mse(val_pred, Y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in psi_model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 100 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:>4}/{epochs}: Loss={total_loss:.6f}  "
                  f"Val={val_loss:.6f}  RelEnt={epoch_rent:.4f}  "
                  f"MonoPen={epoch_mono:.4f}  LR={lr_now:.2e}")

        if patience_cnt >= 200:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    psi_model.load_state_dict(best_state)
    psi_model.to('cpu')
    teacher.to('cpu')
    print(f"  SwiGLU Ψ-model trained. Best val loss: {best_val_loss:.6f}")
    return psi_model


# ─────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────

def evaluate_all(model, X_test, Y_test_orig, scaler_Y, label="Model"):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test), seq_len=NUM_STEPS).numpy()
    pred_orig = inverse_transform_values(pred_scaled, scaler_Y)

    var_names = ['KL', 'KR', 'KLR']
    m_overall = compute_metrics(Y_test_orig, pred_orig)
    m_per_var = {n: compute_metrics(Y_test_orig[:, :, i], pred_orig[:, :, i])
                 for i, n in enumerate(var_names)}
    m_per_slot = []
    for s in range(NUM_STEPS):
        sm = compute_metrics(Y_test_orig[:, s, :], pred_orig[:, s, :])
        sm['per_variable'] = {n: compute_metrics(Y_test_orig[:, s, i:i+1], pred_orig[:, s, i:i+1])
                               for i, n in enumerate(var_names)}
        sm['slot'] = s + 1
        sm['type'] = 'initial' if s == 0 else 'drop'
        m_per_slot.append(sm)

    metrics = {'overall': m_overall, 'per_variable': m_per_var, 'per_slot': m_per_slot}
    print(f"\n  {label}:")
    print(f"    Overall: R²={m_overall['r2']:.4f}  RMSE={m_overall['rmse']:.4e}  MAE={m_overall['mae']:.4e}")
    for n in var_names:
        m = m_per_var[n]
        print(f"    {n:>3}: R²={m['r2']:.4f}  RMSE={m['rmse']:.4e}  MAE={m['mae']:.4e}")
    return metrics, pred_orig


# ─────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("M9: SwiGLU Ψ-NN — Pile Stiffness Degradation")
    print("=" * 60)
    print(f"  Full 3-stage pipeline (A → B → C)")
    print(f"  SwiGLU dim   : {SWIGLU_DIM}  (gate + value branches)")

    # ── Load data ──
    excel_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'M8', 'REAL DATA.xlsx')
    if not os.path.exists(excel_path):
        excel_path = os.path.join(SCRIPT_DIR, 'REAL DATA.xlsx')
    if not os.path.exists(excel_path):
        # try M6 as fallback
        excel_path = os.path.join(M6_DIR, 'REAL DATA.xlsx')
    X_list, Y_list, input_cols, output_cols = load_and_group_data(excel_path)
    Y_array = np.array(Y_list)
    X_array = np.array(X_list)

    scaler_X = RobustScaler()
    X_scaled = scaler_X.fit_transform(X_array)

    Y_sign   = np.sign(Y_array)
    Y_log    = Y_sign * np.log1p(np.abs(Y_array))
    scaler_Y = RobustScaler()
    Y_flat   = Y_log.reshape(-1, 3)
    Y_scaled = scaler_Y.fit_transform(Y_flat).reshape(Y_log.shape)

    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    Y_train, Y_test = Y_scaled[train_idx], Y_scaled[test_idx]
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}, Steps: {NUM_STEPS}")

    # ── Load M6 Teacher (frozen) ──
    print(f"\nLoading M6 teacher from {M6_DIR}...")
    teacher = SlotAttentionDegradation(
        input_size=X_train.shape[1], d_model=64, num_heads=4,
        num_slots=NUM_STEPS, max_seq_len=NUM_STEPS, dropout=0.1, num_iterations=3)
    teacher.load_state_dict(torch.load(
        os.path.join(M6_DIR, 'pile_model.pth'), map_location='cpu', weights_only=True))
    teacher.to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params (frozen)")

    # ── Stage A: Train Student ──
    student = stage_a_distillation(
        teacher, X_train, X_test, Y_train, Y_test,
        epochs=2000, batch_size=8, lr=0.001, mu=0.01)

    # ── Stage B: Structure Discovery ──
    k_star, R, centroids, discovery = stage_b_structure_discovery(
        student, teacher, X_scaled)

    # ── Stage C: Train SwiGLU Ψ-Model ──
    psi_model = stage_c_swiglu_training(
        teacher, k_star, R, centroids,
        X_train, X_test, Y_train, Y_test,
        epochs=2500, batch_size=8, lr=0.001)

    # ── Evaluation ──
    Y_test_orig = Y_array[test_idx]
    print(f"\n{'='*60}")
    print("FINAL COMPARISON (original scale)")
    print(f"{'='*60}")

    teacher_metrics, _ = evaluate_all(teacher, X_test, Y_test_orig, scaler_Y, "M6 Teacher")
    student_metrics, _ = evaluate_all(student, X_test, Y_test_orig, scaler_Y, "Stage-A Student (M9)")
    psi_metrics, _     = evaluate_all(psi_model, X_test, Y_test_orig, scaler_Y, "SwiGLU Ψ-Model (M9)")

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    n_psi     = sum(p.numel() for p in psi_model.parameters())
    print(f"\n  Parameter count:")
    print(f"    M6 Teacher       : {n_teacher:>8,}")
    print(f"    Stage-A Student  : {n_student:>8,}  (SwiGLU MLP)")
    print(f"    M9 SwiGLU Ψ-NN  : {n_psi:>8,}  ({(1-n_psi/n_teacher)*100:.1f}% fewer than M6)")

    R_learned = psi_model.get_relation_matrix().detach().numpy()
    print(f"\n  Learned R  — max-per-row avg : {R_learned.max(axis=1).mean():.3f}")
    print(f"             — sparsity (<0.05): {(R_learned < 0.05).sum()}/{R_learned.size}")

    # ── Save Artifacts ──
    torch.save(psi_model.state_dict(), os.path.join(SCRIPT_DIR, 'pile_model.pth'))
    joblib.dump(scaler_X,    os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_Y,    os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
    joblib.dump(input_cols,  os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
    joblib.dump(NUM_STEPS,   os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
    joblib.dump(psi_metrics, os.path.join(SCRIPT_DIR, 'model_metrics.pkl'))

    R_learned_list = R_learned.tolist()
    psi_config = {
        'k_star': k_star,
        'num_prototypes': k_star,
        'relation_matrix': R_learned_list,
        'relation_matrix_init': R.tolist(),
        'centroids': centroids.tolist(),
        'swiglu_hidden_dim': SWIGLU_DIM,
    }
    joblib.dump(psi_config, os.path.join(SCRIPT_DIR, 'psi_config.pkl'))

    # Save discovery analysis
    with open(os.path.join(SCRIPT_DIR, 'psi_discovery.json'), 'w') as f:
        json.dump(discovery, f, indent=2)

    comparison = {
        'teacher_m6': {
            'params':       n_teacher,
            'overall':      teacher_metrics['overall'],
            'per_variable': teacher_metrics['per_variable'],
        },
        'student_stage_a': {
            'params':       n_student,
            'overall':      student_metrics['overall'],
            'per_variable': student_metrics['per_variable'],
        },
        'psi_model_m7': {   # webapp backward-compat key
            'params':       n_psi,
            'overall':      psi_metrics['overall'],
            'per_variable': psi_metrics['per_variable'],
        },
        'psi_model_m9': {
            'params':       n_psi,
            'overall':      psi_metrics['overall'],
            'per_variable': psi_metrics['per_variable'],
            'k_star':       k_star,
            'compression':  f"{(1 - n_psi / n_teacher) * 100:.1f}%",
            'swiglu_hidden_dim': SWIGLU_DIM,
        },
    }
    with open(os.path.join(SCRIPT_DIR, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    test_data = {
        'X_original': X_array[test_idx],
        'Y_original': Y_array[test_idx],
        'X_scaled':   X_test,
        'input_cols':  input_cols,
        'output_cols': output_cols,
    }
    joblib.dump(test_data, os.path.join(SCRIPT_DIR, 'test_data.pkl'))

    print(f"\n{'='*60}")
    print("All M9 artifacts saved to", SCRIPT_DIR)
    print(f"  pile_model.pth      — SwiGLU Ψ-model weights")
    print(f"  psi_config.pkl      — k*, learned R, SwiGLU config")
    print(f"  psi_discovery.json  — full structure analysis")
    print(f"  comparison.json     — M6 vs Student vs Ψ-model metrics")
    print(f"  test_data.pkl       — test split for webapp")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
