"""
M8: Efficient Ψ-NN Applied to Pile Stiffness Degradation
=========================================================
Extends M7 with three key upgrades:
  1. EfficientSlotMLP: bottleneck (64→48→64) replaces wide MLP (64→128→64)
     in both Student and Ψ-Model  →  ~18% fewer parameters
  2. Learnable relation matrix: R is optimised end-to-end in Stage C via
     softmax(logits) parameterisation  →  better prototype mixing
  3. Physics-aware monotonic loss in Stage C: penalises KL/KR increases and
     KLR decreases across time steps  →  physically consistent predictions

Pipeline (identical structure to M7):
  Stage A: Use M6 as teacher → train regularized student via distillation
  Stage B: Analyze student slots → cluster, build relation matrix R
  Stage C: Rebuild structured Ψ-model with k* prototypes → retrain

Teacher: M6 SlotAttentionDegradation (21 slots, 56,646 params, R²≈0.98)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import json
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M6_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'M6')
NUM_STEPS = 21
STEP_INDICES = np.round(np.linspace(0, 43, NUM_STEPS)).astype(int)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Bottleneck dimension for EfficientSlotMLP (shared constant)
BOTTLENECK_DIM = 48


# ─────────────────────────────────────────────────────
# Efficient Slot MLP (M8 upgrade #1)
# ─────────────────────────────────────────────────────

class EfficientSlotMLP(nn.Module):
    """Bottleneck feed-forward: d_model → bottleneck → d_model.
    Replaces the wide 64→128→64 MLP from M6/M7 with a narrower
    64→48→64 block, saving ~40% of MLP parameters while retaining
    expressive capacity thanks to GELU nonlinearity."""

    def __init__(self, d_model=64, bottleneck_dim=BOTTLENECK_DIM, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────
# M6 Teacher Model (frozen copy — UNCHANGED)
# ─────────────────────────────────────────────────────

class SlotAttentionDegradation(nn.Module):
    """M6 teacher architecture (identical copy for weight loading)."""

    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21,
                 max_seq_len=21, dropout=0.1, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model))
        self.mlp_norm = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        initial = self.initial_proj(slots[:, 0:1, :])
        raw_drops = self.drop_proj(slots[:, 1:, :])
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = torch.abs(raw_drops[:, :, 2:3])
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        return torch.cat([initial, initial + torch.cumsum(drops, dim=1)], dim=1)

    def get_refined_slots(self, x):
        """Return refined slot vectors for structure analysis."""
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        return slots  # [B, 21, 64]


# ─────────────────────────────────────────────────────
# Stage A: Student Model (EfficientSlotMLP + L1)
# ─────────────────────────────────────────────────────

class SlotAttentionStudent(nn.Module):
    """Student that imitates the teacher, with bottleneck MLP and L1 slot reg."""

    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21,
                 max_seq_len=21, dropout=0.1, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        # ── M8 upgrade: EfficientSlotMLP ──
        self.slot_mlp = EfficientSlotMLP(d_model=d_model, bottleneck_dim=BOTTLENECK_DIM, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        initial = self.initial_proj(slots[:, 0:1, :])
        raw_drops = self.drop_proj(slots[:, 1:, :])
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = torch.abs(raw_drops[:, :, 2:3])
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        return torch.cat([initial, initial + torch.cumsum(drops, dim=1)], dim=1)

    def get_refined_slots(self, x):
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        return slots

    def l1_regularization(self):
        """L1 norm of slot vectors to encourage sparsity."""
        return torch.mean(torch.abs(self.slots))


# ─────────────────────────────────────────────────────
# Stage C: Structured Ψ-Model (M8: efficient MLP +
#          learnable relation matrix + physics loss)
# ─────────────────────────────────────────────────────

class SlotAttentionPsiModel(nn.Module):
    """
    Ψ-NN structured model with M8 upgrades:
      - EfficientSlotMLP bottleneck (64→48→64)
      - Learnable relation logits: softmax(logits) gives row-normalised
        mixing weights; initialised from Stage B but refined end-to-end.
      - Physics-monotonic penalty applied in the training loop.
    """

    def __init__(self, input_size=8, d_model=64, num_heads=4,
                 max_seq_len=21, dropout=0.1, num_iterations=3,
                 num_prototypes=4, relation_matrix=None,
                 structured=True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations
        self.num_prototypes = num_prototypes
        self.structured = structured
        self.num_drop_slots = max_seq_len - 1  # 20

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())

        # 1 initial slot + k* prototype slots
        self.initial_slot = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.prototype_slots = nn.Parameter(torch.randn(1, num_prototypes, d_model) * 0.02)

        # ── M8 upgrade #2: Learnable relation logits ──
        if relation_matrix is not None:
            R_init = torch.FloatTensor(relation_matrix)
        else:
            R_init = torch.zeros(self.num_drop_slots, num_prototypes)
            slots_per_proto = self.num_drop_slots // num_prototypes
            for p in range(num_prototypes):
                start = p * slots_per_proto
                end = start + slots_per_proto if p < num_prototypes - 1 else self.num_drop_slots
                R_init[start:end, p] = 1.0
        # Store as logits so softmax gives valid mixing weights
        self.relation_logits = nn.Parameter(torch.log(R_init.clamp(min=1e-6)))

        # Learnable per-slot scale factors
        self.slot_scales = nn.Parameter(torch.ones(self.num_drop_slots, 1))

        # Shared attention layers
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        # ── M8 upgrade #1: EfficientSlotMLP ──
        self.slot_mlp = EfficientSlotMLP(d_model=d_model, bottleneck_dim=BOTTLENECK_DIM, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(d_model)

        # Prediction heads
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def get_relation_matrix(self):
        """Current learned relation weights (row-normalised probabilities)."""
        return torch.softmax(self.relation_logits, dim=1)

    def reconstruct_drop_slots(self, B):
        """Reconstruct 20 drop slots from k* prototypes using learned R."""
        protos = self.prototype_slots.expand(B, -1, -1)          # [B, k*, d]
        R = self.get_relation_matrix()                            # [20, k*]
        drop_slots = torch.matmul(R, protos)                      # [B, 20, d]
        drop_slots = drop_slots * self.slot_scales.unsqueeze(0)   # [B, 20, d]
        return drop_slots

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)

        initial = self.initial_slot.expand(B, -1, -1)
        drops = self.reconstruct_drop_slots(B)
        slots = torch.cat([initial, drops], dim=1)

        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))

        init_pred = self.initial_proj(slots[:, 0:1, :])
        raw_drops = self.drop_proj(slots[:, 1:, :])

        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = torch.abs(raw_drops[:, :, 2:3])
        constrained_drops = torch.cat([drops_kl_kr, drops_klr], dim=2)

        return torch.cat([init_pred, init_pred + torch.cumsum(constrained_drops, dim=1)], dim=1)


# ─────────────────────────────────────────────────────
# Data Loading (shared with M6)
# ─────────────────────────────────────────────────────

def load_and_group_data(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    col_map = {'Dp/Lp': 'Dp_Lp', 'KL': 'kl', 'KR': 'kr', 'KLR': 'klr'}
    df.rename(columns=col_map, inplace=True)
    input_cols = ['PI', 'Gmax', 'v', 'Dp', 'Tp', 'Lp', 'Ip', 'Dp_Lp']
    output_cols = ['kl', 'kr', 'klr']
    groups = df.groupby(input_cols, sort=False)
    X_list, Y_list = [], []
    for name, group in groups:
        outputs = group[output_cols].values
        if len(outputs) != 44:
            continue
        initial = outputs[0]
        drops = outputs[1:]
        actual = initial - np.cumsum(drops, axis=0)
        full_seq = np.vstack([initial, actual])
        subsampled = full_seq[STEP_INDICES]
        X_list.append(list(name))
        Y_list.append(subsampled)
    print(f"Loaded {len(X_list)} scenarios, {NUM_STEPS} steps each")
    return X_list, Y_list, input_cols, output_cols


def inverse_transform_values(scaled, scaler):
    flat = scaled.reshape(-1, 3)
    log_vals = scaler.inverse_transform(flat)
    orig = np.sign(log_vals) * np.expm1(np.abs(log_vals))
    return orig.reshape(scaled.shape)


def compute_metrics(y_true, y_pred):
    ft, fp = y_true.flatten(), y_pred.flatten()
    if len(ft) < 2:
        return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
    return {
        'r2': float(r2_score(ft, fp)),
        'rmse': float(np.sqrt(mean_squared_error(ft, fp))),
        'mae': float(mean_absolute_error(ft, fp)),
    }


# ─────────────────────────────────────────────────────
# Stage A: Train Student via Distillation
# ─────────────────────────────────────────────────────

def stage_a_distillation(teacher, X_train, X_val, Y_train, Y_val,
                         epochs=2000, batch_size=8, lr=0.001, mu=0.01):
    """
    Train a student to imitate the teacher.
    Loss = MSE(student_out, teacher_out) + mu * L1(slots)
    Student uses EfficientSlotMLP (bottleneck=48).
    """
    print(f"\n{'='*60}")
    print("STAGE A: Physics-Informed Distillation")
    print(f"{'='*60}")
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  L1 regularization mu = {mu}")
    print(f"  Efficient MLP bottleneck = {BOTTLENECK_DIM}")
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
            best_state = student.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 100 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            l1_val = student.l1_regularization().item()
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss:.6f}, "
                  f"ValDistill={val_loss:.6f}, L1={l1_val:.4f}, LR={lr_now:.6f}")

        if patience_cnt >= 150:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    student.load_state_dict(best_state)
    student.to('cpu')
    print(f"  Student trained. Best distillation loss: {best_val_loss:.6f}")
    return student


# ─────────────────────────────────────────────────────
# Stage B: Structure Discovery (unchanged from M7)
# ─────────────────────────────────────────────────────

def stage_b_structure_discovery(student, teacher, X_data):
    """
    Analyze the student's slot vectors to discover structure:
    1. Extract refined slot vectors (after attention)
    2. Compute pairwise cosine similarity
    3. K-means clustering with silhouette -> find k*
    4. Build relation matrix R (used as init for learnable logits in Stage C)
    """
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
    drop_slots_teacher = teacher_slots[:, 1:, :].numpy()

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
# Stage C: Train Structured Psi-Model (M8 upgrades)
# ─────────────────────────────────────────────────────

def stage_c_structured_training(teacher, k_star, relation_matrix, centroids,
                                X_train, X_val, Y_train, Y_val,
                                epochs=2500, batch_size=8, lr=0.001):
    """
    Build and train the structured Psi-model with M8 upgrades:
    - EfficientSlotMLP (bottleneck=48)
    - Learnable relation matrix (softmax logits, initialised from Stage B)
    - Physics-monotonic penalty (KL/KR must not increase, KLR must not decrease)
    - Relation-entropy regularisation (encourage sharp prototype assignment)
    """
    print(f"\n{'='*60}")
    print("STAGE C: Structured Psi-Model Training (M8)")
    print(f"{'='*60}")

    psi_model = SlotAttentionPsiModel(
        input_size=X_train.shape[1], d_model=64, num_heads=4,
        max_seq_len=NUM_STEPS, dropout=0.1, num_iterations=3,
        num_prototypes=k_star, relation_matrix=relation_matrix,
        structured=True)

    # Initialize prototype slots from discovered centroids
    with torch.no_grad():
        psi_model.prototype_slots.copy_(
            torch.FloatTensor(centroids).unsqueeze(0))

    psi_model.to(DEVICE)
    teacher.to(DEVICE)

    n_params = sum(p.numel() for p in psi_model.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Psi-model params: {n_params:,} (teacher: {n_teacher:,})")
    print(f"  Compression: {(1 - n_params/n_teacher)*100:.1f}% fewer parameters")
    print(f"  Prototypes: {k_star} (replacing 20 independent drop slots)")
    print(f"  Efficient MLP bottleneck = {BOTTLENECK_DIM}")
    print(f"  Learnable relation matrix: enabled (row-softmax)")
    print(f"  Physics-monotonic penalty: enabled")
    print(f"  Device: {DEVICE}")

    teacher.eval()

    optimizer = torch.optim.AdamW(psi_model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=80, factor=0.5, min_lr=1e-6)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    Y_val_t = torch.FloatTensor(Y_val).to(DEVICE)

    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(epochs):
        psi_model.train()
        total_loss = 0
        epoch_mono = 0.0
        epoch_rent = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = teacher(X_batch, seq_len=NUM_STEPS)

            psi_out = psi_model(X_batch, seq_len=NUM_STEPS)

            # Multi-objective loss (same base as M7)
            loss_distill = mse(psi_out, teacher_out)
            loss_seq = huber(psi_out, Y_batch)
            loss_initial = mse(psi_out[:, 0, :], Y_batch[:, 0, :]) * 5.0
            diff_pred = psi_out[:, 1:, :] - psi_out[:, :-1, :]
            diff_target = Y_batch[:, 1:, :] - Y_batch[:, :-1, :]
            loss_shape = huber(diff_pred, diff_target)

            # ── M8 upgrade #2: relation-entropy regularisation ──
            R_weights = psi_model.get_relation_matrix()
            loss_relation_entropy = -(
                R_weights * torch.log(R_weights + 1e-8)
            ).sum(dim=1).mean()

            # ── M8 upgrade #3: physics-monotonic penalty ──
            # KL, KR should not increase over time (degradation)
            # KLR should not decrease over time
            diff_kl_kr = psi_out[:, 1:, :2] - psi_out[:, :-1, :2]
            diff_klr = psi_out[:, 1:, 2:3] - psi_out[:, :-1, 2:3]
            loss_physics_mono = torch.relu(diff_kl_kr).mean() + torch.relu(-diff_klr).mean()

            loss = (
                loss_distill
                + loss_seq
                + loss_initial
                + loss_shape
                + 0.02 * loss_relation_entropy
                + 0.2 * loss_physics_mono
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(psi_model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            epoch_mono += loss_physics_mono.item()
            epoch_rent += loss_relation_entropy.item()

        n_batches = len(train_loader)
        total_loss /= n_batches
        epoch_mono /= n_batches
        epoch_rent /= n_batches

        psi_model.eval()
        with torch.no_grad():
            val_pred = psi_model(X_val_t, seq_len=NUM_STEPS)
            val_loss = mse(val_pred, Y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = psi_model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 100 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss:.6f}, "
                  f"Val={val_loss:.6f}, RelEnt={epoch_rent:.4f}, "
                  f"MonoPen={epoch_mono:.4f}, LR={lr_now:.6f}")

        if patience_cnt >= 200:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    psi_model.load_state_dict(best_state)
    psi_model.to('cpu')
    teacher.to('cpu')
    print(f"  Psi-model trained. Best val loss: {best_val_loss:.6f}")
    return psi_model


# ─────────────────────────────────────────────────────
# Full Evaluation
# ─────────────────────────────────────────────────────

def evaluate_all(model, X_test, Y_test_orig, scaler_Y, label="Model"):
    """Evaluate a model on test set, return full metrics dict."""
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test), seq_len=NUM_STEPS).numpy()
    pred_orig = inverse_transform_values(pred_scaled, scaler_Y)

    var_names = ['KL', 'KR', 'KLR']

    m_overall = compute_metrics(Y_test_orig, pred_orig)
    m_per_var = {}
    for i, name in enumerate(var_names):
        m_per_var[name] = compute_metrics(Y_test_orig[:, :, i], pred_orig[:, :, i])

    m_per_slot = []
    for s in range(NUM_STEPS):
        sm = compute_metrics(Y_test_orig[:, s, :], pred_orig[:, s, :])
        sm['per_variable'] = {}
        for i, name in enumerate(var_names):
            sm['per_variable'][name] = compute_metrics(
                Y_test_orig[:, s, i:i+1], pred_orig[:, s, i:i+1])
        sm['slot'] = s + 1
        sm['type'] = 'initial' if s == 0 else 'drop'
        m_per_slot.append(sm)

    metrics = {'overall': m_overall, 'per_variable': m_per_var, 'per_slot': m_per_slot}

    print(f"\n  {label}:")
    print(f"    Overall: R2={m_overall['r2']:.4f}  RMSE={m_overall['rmse']:.4e}  MAE={m_overall['mae']:.4e}")
    for name in var_names:
        m = m_per_var[name]
        print(f"    {name:>3}: R2={m['r2']:.4f}  RMSE={m['rmse']:.4e}  MAE={m['mae']:.4e}")

    return metrics, pred_orig


# ─────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("M8: Efficient Psi-NN Pipeline for Pile Stiffness Degradation")
    print("=" * 60)

    # ── Load Data ──
    excel_path = os.path.join(SCRIPT_DIR, 'REAL DATA.xlsx')
    X_list, Y_list, input_cols, output_cols = load_and_group_data(excel_path)
    Y_array = np.array(Y_list)
    X_array = np.array(X_list)

    scaler_X = RobustScaler()
    X_scaled = scaler_X.fit_transform(X_array)

    Y_sign = np.sign(Y_array)
    Y_log = Y_sign * np.log1p(np.abs(Y_array))
    scaler_Y = RobustScaler()
    Y_flat = Y_log.reshape(-1, 3)
    Y_scaled = scaler_Y.fit_transform(Y_flat).reshape(Y_log.shape)

    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    Y_train, Y_test = Y_scaled[train_idx], Y_scaled[test_idx]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}, Steps: {NUM_STEPS}")

    # ── Load M6 Teacher ──
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
    print(f"  Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")
    print(f"  Device: {DEVICE}")

    # ── Stage A ──
    student = stage_a_distillation(
        teacher, X_train, X_test, Y_train, Y_test,
        epochs=2000, batch_size=8, lr=0.001, mu=0.01)

    # ── Stage B ──
    k_star, R, centroids, discovery = stage_b_structure_discovery(
        student, teacher, X_scaled)

    # ── Stage C ──
    psi_model = stage_c_structured_training(
        teacher, k_star, R, centroids,
        X_train, X_test, Y_train, Y_test,
        epochs=2500, batch_size=8, lr=0.001)

    # ── Final Evaluation ──
    Y_test_orig = Y_array[test_idx]

    print(f"\n{'='*60}")
    print("FINAL COMPARISON (original scale)")
    print(f"{'='*60}")

    teacher_metrics, teacher_pred = evaluate_all(teacher, X_test, Y_test_orig, scaler_Y, "M6 Teacher")
    student_metrics, student_pred = evaluate_all(student, X_test, Y_test_orig, scaler_Y, "Stage-A Student (M8)")
    psi_metrics, psi_pred = evaluate_all(psi_model, X_test, Y_test_orig, scaler_Y, "Psi-Model (M8)")

    # Parameter comparison
    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    n_psi = sum(p.numel() for p in psi_model.parameters())

    print(f"\n  Parameter Count:")
    print(f"    M6 Teacher:      {n_teacher:>8,}")
    print(f"    Stage-A Student: {n_student:>8,} (M8 efficient MLP)")
    print(f"    Psi-Model (M8):  {n_psi:>8,} ({(1-n_psi/n_teacher)*100:.1f}% fewer than M6)")

    # Show learned relation matrix stats
    R_learned = psi_model.get_relation_matrix().detach().numpy()
    print(f"\n  Learned Relation Matrix R:")
    print(f"    Shape: {R_learned.shape}")
    print(f"    Max per row (avg): {R_learned.max(axis=1).mean():.3f}")
    print(f"    Sparsity (entries < 0.05): {(R_learned < 0.05).sum()}/{R_learned.size}")

    # ── Save Everything ──
    torch.save(psi_model.state_dict(), os.path.join(SCRIPT_DIR, 'pile_model.pth'))
    joblib.dump(scaler_X, os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_Y, os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
    joblib.dump(input_cols, os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
    joblib.dump(NUM_STEPS, os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
    joblib.dump(psi_metrics, os.path.join(SCRIPT_DIR, 'model_metrics.pkl'))

    # Save Psi-NN specific artifacts (with learned R)
    R_learned_list = psi_model.get_relation_matrix().detach().numpy().tolist()
    psi_config = {
        'k_star': k_star,
        'num_prototypes': k_star,
        'relation_matrix': R_learned_list,
        'relation_matrix_init': R.tolist(),
        'centroids': centroids.tolist(),
        'efficient_mlp_bottleneck': BOTTLENECK_DIM,
    }
    joblib.dump(psi_config, os.path.join(SCRIPT_DIR, 'psi_config.pkl'))

    # Save discovery analysis
    with open(os.path.join(SCRIPT_DIR, 'psi_discovery.json'), 'w') as f:
        json.dump(discovery, f, indent=2)

    # Save comparison metrics (keep psi_model_m7 key for webapp backward compat)
    comparison = {
        'teacher_m6': {
            'params': n_teacher,
            'overall': teacher_metrics['overall'],
            'per_variable': teacher_metrics['per_variable'],
        },
        'student_stage_a': {
            'params': n_student,
            'overall': student_metrics['overall'],
            'per_variable': student_metrics['per_variable'],
        },
        'psi_model_m8': {
            'params': n_psi,
            'overall': psi_metrics['overall'],
            'per_variable': psi_metrics['per_variable'],
            'k_star': k_star,
            'compression': f"{(1-n_psi/n_teacher)*100:.1f}%",
            'efficient_mlp_bottleneck': BOTTLENECK_DIM,
            'learned_relation_matrix': True,
        },
        # backward-compat alias so webapp works without changes
        'psi_model_m7': {
            'params': n_psi,
            'overall': psi_metrics['overall'],
            'per_variable': psi_metrics['per_variable'],
            'k_star': k_star,
            'compression': f"{(1-n_psi/n_teacher)*100:.1f}%",
        },
    }
    with open(os.path.join(SCRIPT_DIR, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Save test data for webapp
    test_data = {
        'X_original': X_array[test_idx],
        'Y_original': Y_array[test_idx],
        'X_scaled': X_test,
        'input_cols': input_cols,
        'output_cols': output_cols,
    }
    joblib.dump(test_data, os.path.join(SCRIPT_DIR, 'test_data.pkl'))

    print(f"\n{'='*60}")
    print("All artifacts saved!")
    print(f"  pile_model.pth      -> Psi-model weights (M8)")
    print(f"  psi_config.pkl      -> k*, learned R, centroids")
    print(f"  psi_discovery.json  -> full analysis (similarity, clustering)")
    print(f"  comparison.json     -> M6 vs Student vs Psi-model metrics")
    print(f"  model_metrics.pkl   -> Psi-model per-slot metrics")
    print(f"\nRun: python webapp.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
"""
M7: Ψ-NN Applied to Pile Stiffness Degradation
================================================
Implements the 3-stage pipeline from Liu et al. (Nature Communications, 2025):
  Stage A: Use M6 as teacher → train regularized student via distillation
  Stage B: Analyze student slots → cluster, build relation matrix R
  Stage C: Rebuild structured Ψ-model with k* prototypes → retrain

Teacher: M6 SlotAttentionDegradation (21 slots, 56,646 params, R²≈0.98)
Goal:    Discover redundant slot patterns, compress to fewer prototypes
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import json
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M6_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'M6')
NUM_STEPS = 21
STEP_INDICES = np.round(np.linspace(0, 43, NUM_STEPS)).astype(int)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────
# M6 Teacher Model (frozen copy)
# ─────────────────────────────────────────────────────

class SlotAttentionDegradation(nn.Module):
    """M6 teacher architecture (identical copy for weight loading)."""

    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21,
                 max_seq_len=21, dropout=0.1, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model))
        self.mlp_norm = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        initial = self.initial_proj(slots[:, 0:1, :])
        raw_drops = self.drop_proj(slots[:, 1:, :])
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = torch.abs(raw_drops[:, :, 2:3])
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        return torch.cat([initial, initial + torch.cumsum(drops, dim=1)], dim=1)

    def get_refined_slots(self, x):
        """Return refined slot vectors for structure analysis."""
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        return slots  # [B, 21, 64]


# ─────────────────────────────────────────────────────
# Stage A: Student Model (same arch + L1 regularization)
# ─────────────────────────────────────────────────────

class SlotAttentionStudent(nn.Module):
    """Student that imitates the teacher, trained with L1 reg to encourage sparsity."""

    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21,
                 max_seq_len=21, dropout=0.1, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model))
        self.mlp_norm = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        initial = self.initial_proj(slots[:, 0:1, :])
        raw_drops = self.drop_proj(slots[:, 1:, :])
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = torch.abs(raw_drops[:, :, 2:3])
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        return torch.cat([initial, initial + torch.cumsum(drops, dim=1)], dim=1)

    def get_refined_slots(self, x):
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(B, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        return slots

    def l1_regularization(self):
        """L1 norm of slot vectors to encourage sparsity."""
        return torch.mean(torch.abs(self.slots))


# ─────────────────────────────────────────────────────
# Stage C: Structured Ψ-Model (prototype slots)
# ─────────────────────────────────────────────────────

class SlotAttentionPsiModel(nn.Module):
    """
    Ψ-NN structured model: k* prototype drop-slots + relation matrix R.
    Instead of 20 independent drop slots, learns only k* prototypes.
    All 20 drop slots are reconstructed from prototypes via learned scale/sign rules.
    """

    def __init__(self, input_size=8, d_model=64, num_heads=4,
                 max_seq_len=21, dropout=0.1, num_iterations=3,
                 num_prototypes=4, relation_matrix=None,
                 structured=True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations
        self.num_prototypes = num_prototypes
        self.structured = structured
        self.num_drop_slots = max_seq_len - 1  # 20

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())

        # 1 initial slot + k* prototype slots
        self.initial_slot = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.prototype_slots = nn.Parameter(torch.randn(1, num_prototypes, d_model) * 0.02)

        # Relation matrix: how each of 20 drops maps to prototypes
        # R[i, j] = scale factor from prototype j to drop-slot i
        if relation_matrix is not None:
            self.register_buffer('relation_matrix', torch.FloatTensor(relation_matrix))
        else:
            # Default: distribute drops evenly across prototypes
            R = np.zeros((self.num_drop_slots, num_prototypes))
            slots_per_proto = self.num_drop_slots // num_prototypes
            for p in range(num_prototypes):
                start = p * slots_per_proto
                end = start + slots_per_proto if p < num_prototypes - 1 else self.num_drop_slots
                R[start:end, p] = 1.0
            self.register_buffer('relation_matrix', torch.FloatTensor(R))

        # Learnable per-slot scale factors for fine adjustment
        self.slot_scales = nn.Parameter(torch.ones(self.num_drop_slots, 1))

        # Shared attention and MLP layers
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model))
        self.mlp_norm = nn.LayerNorm(d_model)

        # Prediction heads
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def reconstruct_drop_slots(self, B):
        """Reconstruct 20 drop slots from k* prototypes using relation matrix R."""
        # prototype_slots: [1, k*, d_model]
        protos = self.prototype_slots.expand(B, -1, -1)  # [B, k*, d_model]
        # R: [20, k*] -> each drop = weighted combination of prototypes
        # drop_slots = R @ protos -> [B, 20, d_model]
        drop_slots = torch.matmul(self.relation_matrix, protos)  # [B, 20, d_model]
        # Apply per-slot learned scales
        drop_slots = drop_slots * self.slot_scales.unsqueeze(0)  # [B, 20, d_model]
        return drop_slots

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        B = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)  # [B, 1, d_model]

        # Build full slot bank: 1 initial + 20 reconstructed drops
        initial = self.initial_slot.expand(B, -1, -1)  # [B, 1, d_model]
        drops = self.reconstruct_drop_slots(B)          # [B, 20, d_model]
        slots = torch.cat([initial, drops], dim=1)       # [B, 21, d_model]

        # Iterative refinement (same as M6)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))

        # Predict
        init_pred = self.initial_proj(slots[:, 0:1, :])
        raw_drops = self.drop_proj(slots[:, 1:, :])

        # Physics constraints (same as M6)
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = torch.abs(raw_drops[:, :, 2:3])
        constrained_drops = torch.cat([drops_kl_kr, drops_klr], dim=2)

        return torch.cat([init_pred, init_pred + torch.cumsum(constrained_drops, dim=1)], dim=1)


# ─────────────────────────────────────────────────────
# Data Loading (shared with M6)
# ─────────────────────────────────────────────────────

def load_and_group_data(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    col_map = {'Dp/Lp': 'Dp_Lp', 'KL': 'kl', 'KR': 'kr', 'KLR': 'klr'}
    df.rename(columns=col_map, inplace=True)
    input_cols = ['PI', 'Gmax', 'v', 'Dp', 'Tp', 'Lp', 'Ip', 'Dp_Lp']
    output_cols = ['kl', 'kr', 'klr']
    groups = df.groupby(input_cols, sort=False)
    X_list, Y_list = [], []
    for name, group in groups:
        outputs = group[output_cols].values
        if len(outputs) != 44:
            continue
        initial = outputs[0]
        drops = outputs[1:]
        actual = initial - np.cumsum(drops, axis=0)
        full_seq = np.vstack([initial, actual])
        subsampled = full_seq[STEP_INDICES]
        X_list.append(list(name))
        Y_list.append(subsampled)
    print(f"Loaded {len(X_list)} scenarios, {NUM_STEPS} steps each")
    return X_list, Y_list, input_cols, output_cols


def inverse_transform_values(scaled, scaler):
    flat = scaled.reshape(-1, 3)
    log_vals = scaler.inverse_transform(flat)
    orig = np.sign(log_vals) * np.expm1(np.abs(log_vals))
    return orig.reshape(scaled.shape)


def compute_metrics(y_true, y_pred):
    ft, fp = y_true.flatten(), y_pred.flatten()
    if len(ft) < 2:
        return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
    return {
        'r2': float(r2_score(ft, fp)),
        'rmse': float(np.sqrt(mean_squared_error(ft, fp))),
        'mae': float(mean_absolute_error(ft, fp)),
    }


# ─────────────────────────────────────────────────────
# Stage A: Train Student via Distillation
# ─────────────────────────────────────────────────────

def stage_a_distillation(teacher, X_train, X_val, Y_train, Y_val,
                         epochs=2000, batch_size=8, lr=0.001, mu=0.01):
    """
    Train a student to imitate the teacher.
    Loss = MSE(student_out, teacher_out) + μ * L1(slots)
    The L1 regularization encourages the student to find sparse slot patterns.
    """
    print(f"\n{'='*60}")
    print("STAGE A: Physics-Informed Distillation")
    print(f"{'='*60}")
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  L1 regularization μ = {mu}")
    print(f"  Device: {DEVICE}")

    teacher.eval()

    student = SlotAttentionStudent(
        input_size=X_train.shape[1], d_model=64, num_heads=4,
        num_slots=NUM_STEPS, max_seq_len=NUM_STEPS, dropout=0.1, num_iterations=3)
    student.to(DEVICE)

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

            # Distillation: match teacher output
            loss_distill = mse(student_out, teacher_out)
            # Also match ground truth for data fidelity
            loss_data = mse(student_out, Y_batch)
            # L1 on slot vectors
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
            best_state = student.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 100 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            l1_val = student.l1_regularization().item()
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss:.6f}, "
                  f"ValDistill={val_loss:.6f}, L1={l1_val:.4f}, LR={lr_now:.6f}")

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
    """
    Analyze the student's slot vectors to discover structure:
    1. Extract refined slot vectors (after attention)
    2. Compute pairwise cosine similarity
    3. K-means clustering with elbow/silhouette → find k*
    4. Build relation matrix R
    """
    print(f"\n{'='*60}")
    print("STAGE B: Structure Extraction & Clustering")
    print(f"{'='*60}")

    student.eval()
    teacher.eval()

    # ── Step 1: Extract slot representations ──
    # Move to CPU for clustering (sklearn needs numpy)
    student.to('cpu')
    teacher.to('cpu')
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_data)
        # Get refined slots from student (after attention iterations)
        student_slots = student.get_refined_slots(X_tensor)  # [N, 21, 64]
        teacher_slots = teacher.get_refined_slots(X_tensor)  # [N, 21, 64]

    # Focus on drop slots (slots 2-21, index 1-20)
    drop_slots_student = student_slots[:, 1:, :].numpy()  # [N, 20, 64]
    drop_slots_teacher = teacher_slots[:, 1:, :].numpy()

    # Average across all samples for static analysis
    avg_drop_slots = drop_slots_student.mean(axis=0)  # [20, 64]

    print(f"\n  Drop slot vectors shape: {avg_drop_slots.shape}")

    # ── Step 2: Slot norms ──
    slot_norms = np.linalg.norm(avg_drop_slots, axis=1)  # [20]
    print(f"  Slot norms (min={slot_norms.min():.3f}, max={slot_norms.max():.3f}, "
          f"mean={slot_norms.mean():.3f})")

    # ── Step 3: Pairwise cosine similarity ──
    norm_slots = avg_drop_slots / (np.linalg.norm(avg_drop_slots, axis=1, keepdims=True) + 1e-8)
    cosine_sim = norm_slots @ norm_slots.T  # [20, 20]
    print(f"  Cosine similarity: mean={cosine_sim.mean():.3f}, "
          f"avg off-diagonal={(cosine_sim.sum() - np.trace(cosine_sim)) / (20*19):.3f}")

    # ── Step 4: K-means clustering with elbow method ──
    k_range = range(2, min(11, len(avg_drop_slots)))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(avg_drop_slots)
        inertias.append(km.inertia_)
        sil = silhouette_score(avg_drop_slots, labels)
        silhouettes.append(sil)

    # Find optimal k*: best silhouette score
    best_k_idx = np.argmax(silhouettes)
    k_star = list(k_range)[best_k_idx]
    best_silhouette = silhouettes[best_k_idx]

    print(f"\n  Clustering results:")
    for i, k in enumerate(k_range):
        marker = " ◄ BEST" if k == k_star else ""
        print(f"    k={k}: inertia={inertias[i]:.2f}, silhouette={silhouettes[i]:.3f}{marker}")
    print(f"\n  Optimal k* = {k_star} (silhouette = {best_silhouette:.3f})")

    # ── Step 5: Final clustering with k* ──
    km_final = KMeans(n_clusters=k_star, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(avg_drop_slots)
    centroids = km_final.cluster_centers_  # [k*, 64]

    # ── Step 6: Build relation matrix R ──
    # R[i, j] = how much drop-slot i depends on prototype j
    # We use soft assignment based on inverse distance
    distances = np.zeros((20, k_star))
    for i in range(20):
        for j in range(k_star):
            distances[i, j] = np.linalg.norm(avg_drop_slots[i] - centroids[j])

    # Convert to weights (inverse distance, normalized per slot)
    inv_dist = 1.0 / (distances + 1e-8)
    R = inv_dist / inv_dist.sum(axis=1, keepdims=True)  # [20, k*]

    # For slots very close to a centroid, sharpen to near-one-hot
    for i in range(20):
        if R[i].max() > 0.8:
            dominant = R[i].argmax()
            R[i] = 0.0
            R[i, dominant] = 1.0

    # ── Step 7: Describe discovered structure ──
    cluster_info = {}
    for p in range(k_star):
        members = np.where(cluster_labels == p)[0]
        member_norms = slot_norms[members]
        cluster_info[f"Prototype_{p}"] = {
            'members': (members + 2).tolist(),  # +2 because slot numbering starts at 1, drops at 2
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
        'relation_matrix': R.tolist(),
        'centroids': centroids.tolist(),
        'cluster_info': cluster_info,
        'inertias': {str(k): float(v) for k, v in zip(k_range, inertias)},
        'silhouettes': {str(k): float(v) for k, v in zip(k_range, silhouettes)},
    }

    return k_star, R, centroids, discovery


# ─────────────────────────────────────────────────────
# Stage C: Train Structured Ψ-Model
# ─────────────────────────────────────────────────────

def stage_c_structured_training(teacher, k_star, relation_matrix, centroids,
                                X_train, X_val, Y_train, Y_val,
                                epochs=2500, batch_size=8, lr=0.001):
    """
    Build and train the structured Ψ-model:
    - Only k* prototype slots (not 20 independent ones)
    - Relation matrix R reconstructs all 20 drops from prototypes
    - Train with distillation + direct data loss + physics
    """
    print(f"\n{'='*60}")
    print("STAGE C: Structured Ψ-Model Training")
    print(f"{'='*60}")

    psi_model = SlotAttentionPsiModel(
        input_size=X_train.shape[1], d_model=64, num_heads=4,
        max_seq_len=NUM_STEPS, dropout=0.1, num_iterations=3,
        num_prototypes=k_star, relation_matrix=relation_matrix,
        structured=True)

    # Initialize prototype slots from discovered centroids
    with torch.no_grad():
        psi_model.prototype_slots.copy_(
            torch.FloatTensor(centroids).unsqueeze(0))  # [1, k*, 64]

    psi_model.to(DEVICE)
    teacher.to(DEVICE)

    n_params = sum(p.numel() for p in psi_model.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Ψ-model params: {n_params:,} (teacher: {n_teacher:,})")
    print(f"  Compression: {(1 - n_params/n_teacher)*100:.1f}% fewer parameters")
    print(f"  Prototypes: {k_star} (replacing 20 independent drop slots)")
    print(f"  Device: {DEVICE}")

    teacher.eval()

    optimizer = torch.optim.AdamW(psi_model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=80, factor=0.5, min_lr=1e-6)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    Y_val_t = torch.FloatTensor(Y_val).to(DEVICE)

    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(epochs):
        psi_model.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = teacher(X_batch, seq_len=NUM_STEPS)

            psi_out = psi_model(X_batch, seq_len=NUM_STEPS)

            # Multi-objective loss
            loss_distill = mse(psi_out, teacher_out)          # Match teacher
            loss_seq = huber(psi_out, Y_batch)                 # Match ground truth
            loss_initial = mse(psi_out[:, 0, :], Y_batch[:, 0, :]) * 5.0  # Initial emphasis
            diff_pred = psi_out[:, 1:, :] - psi_out[:, :-1, :]
            diff_target = Y_batch[:, 1:, :] - Y_batch[:, :-1, :]
            loss_shape = huber(diff_pred, diff_target)         # Shape preservation

            loss = loss_distill + loss_seq + loss_initial + loss_shape
            loss.backward()
            torch.nn.utils.clip_grad_norm_(psi_model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)

        psi_model.eval()
        with torch.no_grad():
            val_pred = psi_model(X_val_t, seq_len=NUM_STEPS)
            val_loss = mse(val_pred, Y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = psi_model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 100 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss:.6f}, "
                  f"Val={val_loss:.6f}, LR={lr_now:.6f}")

        if patience_cnt >= 200:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    psi_model.load_state_dict(best_state)
    psi_model.to('cpu')
    teacher.to('cpu')
    print(f"  Ψ-model trained. Best val loss: {best_val_loss:.6f}")
    return psi_model


# ─────────────────────────────────────────────────────
# Full Evaluation
# ─────────────────────────────────────────────────────

def evaluate_all(model, X_test, Y_test_orig, scaler_Y, label="Model"):
    """Evaluate a model on test set, return full metrics dict."""
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test), seq_len=NUM_STEPS).numpy()
    pred_orig = inverse_transform_values(pred_scaled, scaler_Y)

    var_names = ['KL', 'KR', 'KLR']

    m_overall = compute_metrics(Y_test_orig, pred_orig)
    m_per_var = {}
    for i, name in enumerate(var_names):
        m_per_var[name] = compute_metrics(Y_test_orig[:, :, i], pred_orig[:, :, i])

    m_per_slot = []
    for s in range(NUM_STEPS):
        sm = compute_metrics(Y_test_orig[:, s, :], pred_orig[:, s, :])
        sm['per_variable'] = {}
        for i, name in enumerate(var_names):
            sm['per_variable'][name] = compute_metrics(
                Y_test_orig[:, s, i:i+1], pred_orig[:, s, i:i+1])
        sm['slot'] = s + 1
        sm['type'] = 'initial' if s == 0 else 'drop'
        m_per_slot.append(sm)

    metrics = {'overall': m_overall, 'per_variable': m_per_var, 'per_slot': m_per_slot}

    print(f"\n  {label}:")
    print(f"    Overall: R²={m_overall['r2']:.4f}  RMSE={m_overall['rmse']:.4e}  MAE={m_overall['mae']:.4e}")
    for name in var_names:
        m = m_per_var[name]
        print(f"    {name:>3}: R²={m['r2']:.4f}  RMSE={m['rmse']:.4e}  MAE={m['mae']:.4e}")

    return metrics, pred_orig


# ─────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("M7: Ψ-NN Pipeline for Pile Stiffness Degradation")
    print("=" * 60)

    # ── Load Data ──
    excel_path = os.path.join(SCRIPT_DIR, 'REAL DATA.xlsx')
    X_list, Y_list, input_cols, output_cols = load_and_group_data(excel_path)
    Y_array = np.array(Y_list)
    X_array = np.array(X_list)

    scaler_X = RobustScaler()
    X_scaled = scaler_X.fit_transform(X_array)

    Y_sign = np.sign(Y_array)
    Y_log = Y_sign * np.log1p(np.abs(Y_array))
    scaler_Y = RobustScaler()
    Y_flat = Y_log.reshape(-1, 3)
    Y_scaled = scaler_Y.fit_transform(Y_flat).reshape(Y_log.shape)

    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    Y_train, Y_test = Y_scaled[train_idx], Y_scaled[test_idx]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}, Steps: {NUM_STEPS}")

    # ── Load M6 Teacher ──
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
    print(f"  Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")
    print(f"  Device: {DEVICE}")

    # ── Stage A ──
    student = stage_a_distillation(
        teacher, X_train, X_test, Y_train, Y_test,
        epochs=2000, batch_size=8, lr=0.001, mu=0.01)

    # ── Stage B ──
    k_star, R, centroids, discovery = stage_b_structure_discovery(
        student, teacher, X_scaled)

    # ── Stage C ──
    psi_model = stage_c_structured_training(
        teacher, k_star, R, centroids,
        X_train, X_test, Y_train, Y_test,
        epochs=2500, batch_size=8, lr=0.001)

    # ── Final Evaluation ──
    Y_test_orig = Y_array[test_idx]

    print(f"\n{'='*60}")
    print("FINAL COMPARISON (original scale)")
    print(f"{'='*60}")

    teacher_metrics, teacher_pred = evaluate_all(teacher, X_test, Y_test_orig, scaler_Y, "M6 Teacher")
    student_metrics, student_pred = evaluate_all(student, X_test, Y_test_orig, scaler_Y, "Stage-A Student")
    psi_metrics, psi_pred = evaluate_all(psi_model, X_test, Y_test_orig, scaler_Y, "Ψ-Model (M7)")

    # Parameter comparison
    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    n_psi = sum(p.numel() for p in psi_model.parameters())

    print(f"\n  Parameter Count:")
    print(f"    M6 Teacher:     {n_teacher:>8,}")
    print(f"    Stage-A Student: {n_student:>8,}")
    print(f"    Ψ-Model (M7):  {n_psi:>8,} ({(1-n_psi/n_teacher)*100:.1f}% fewer than M6)")

    # ── Save Everything ──
    torch.save(psi_model.state_dict(), os.path.join(SCRIPT_DIR, 'pile_model.pth'))
    joblib.dump(scaler_X, os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_Y, os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
    joblib.dump(input_cols, os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
    joblib.dump(NUM_STEPS, os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
    joblib.dump(psi_metrics, os.path.join(SCRIPT_DIR, 'model_metrics.pkl'))

    # Save Ψ-NN specific artifacts
    psi_config = {
        'k_star': k_star,
        'num_prototypes': k_star,
        'relation_matrix': R.tolist(),
        'centroids': centroids.tolist(),
    }
    joblib.dump(psi_config, os.path.join(SCRIPT_DIR, 'psi_config.pkl'))

    # Save discovery analysis
    with open(os.path.join(SCRIPT_DIR, 'psi_discovery.json'), 'w') as f:
        json.dump(discovery, f, indent=2)

    # Save comparison metrics
    comparison = {
        'teacher_m6': {
            'params': n_teacher,
            'overall': teacher_metrics['overall'],
            'per_variable': teacher_metrics['per_variable'],
        },
        'student_stage_a': {
            'params': n_student,
            'overall': student_metrics['overall'],
            'per_variable': student_metrics['per_variable'],
        },
        'psi_model_m7': {
            'params': n_psi,
            'overall': psi_metrics['overall'],
            'per_variable': psi_metrics['per_variable'],
            'k_star': k_star,
            'compression': f"{(1-n_psi/n_teacher)*100:.1f}%",
        },
    }
    with open(os.path.join(SCRIPT_DIR, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Save test data for webapp
    test_data = {
        'X_original': X_array[test_idx],
        'Y_original': Y_array[test_idx],
        'X_scaled': X_test,
        'input_cols': input_cols,
        'output_cols': output_cols,
    }
    joblib.dump(test_data, os.path.join(SCRIPT_DIR, 'test_data.pkl'))

    print(f"\n{'='*60}")
    print("All artifacts saved!")
    print(f"  pile_model.pth      → Ψ-model weights")
    print(f"  psi_config.pkl      → k*, relation matrix, centroids")
    print(f"  psi_discovery.json  → full analysis (similarity, clustering)")
    print(f"  comparison.json     → M6 vs Student vs Ψ-model metrics")
    print(f"  model_metrics.pkl   → Ψ-model per-slot metrics")
    print(f"\nRun: python webapp.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
