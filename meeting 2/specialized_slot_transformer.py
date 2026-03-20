"""
=============================================================================
Meeting 2 — Specialized-Slot Yield Transformer  (v2)
=============================================================================
Professor's notes:
  - Slots with activation functions ANALOGOUS TO YIELD SURFACES
  - Each slot does something SPECIFIC
  - A FUSION slot combines everything → degradation analysis
  - Data from SEPARATE CSV files per domain
  - SELF-CALIBRATING governing parameters
  - LESS parameters = BETTER

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │  CSV Files (4 domains)                                          │
  │                                                                 │
  │  structural.csv ─→ Slot 0 (Structural)   σ_y, p learned        │
  │  soil.csv       ─→ Slot 1 (Soil)         σ_y, p learned        │
  │  loading.csv    ─→ Slot 2 (Loading)      σ_y, p learned        │
  │  environment.csv─→ Slot 3 (Environment)  σ_y, p learned        │
  │                        ↓  ↓  ↓  ↓                              │
  │                   ┌────────────────┐                            │
  │                   │  FUSION SLOT   │  (Slot 4)                  │
  │                   │  Combines all  │  Cross-attention + yield   │
  │                   └───────┬────────┘                            │
  │                           ↓                                     │
  │                  Transformer Encoder                            │
  │                           ↓                                     │
  │       ┌────────┬──────────┬─────────────┐                       │
  │       u (disp) K (stiff)  G/Gmax (soil) f/f0 (freq shift)     │
  └──────────────────────────────────────────────────────────────────┘

Each specialized slot:
  - Has its OWN embedding layer for its domain features
  - Has its OWN yield activation (σ_y, p learned)
  - Processes ONLY its domain's data

The fusion slot:
  - Takes ALL 4 specialized slot outputs via cross-attention
  - Has its OWN yield activation (broader σ_y for combined response)
  - Produces the final representation for output heads

Outputs (4 targets):
  u  — pile-head displacement [m]
  K  — soil-pile lateral stiffness [kN/m]
  G/Gmax — soil shear-modulus degradation ratio [0-1]
  f/f0   — natural frequency ratio (post/pre typhoon) [0-1]

Sources:
  - Locatello et al. (2020), Object-Centric Learning with Slot Attention
  - Vaswani et al. (2017), Attention Is All You Need
  - Ramberg & Osgood (1943), load-displacement yield model
  - Hardin & Drnevich (1972), G/Gmax degradation
  - Bhattacharya (2019), OWT natural frequency
  - Kondner (1963), hyperbolic soil model
=============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

# ── Config (deliberately small — "less is better") ──────────────────────────
D_MODEL     = 32          # hidden size: 32 numbers represent each time step
N_SPEC_SLOTS = 4          # 4 specialized slots (structural, soil, loading, env)
SLOT_ITERS  = 3           # fusion slot refines its answer 3 times
EPOCHS      = 200         # how many times we go through all training data
BATCH       = 32          # process 32 scenarios at once for efficiency
LR          = 2e-3        # learning rate (how fast the model updates)

# Feature counts per domain CSV (each CSV has 4 numeric columns)
FEAT_STRUCT = 4   # L_pile, D_pile, t_wall, EI           (pile itself)
FEAT_SOIL   = 4   # Su, G0, void_ratio, OCR              (ground)
FEAT_LOAD   = 4   # F_lateral, M_base, N_cycles, freq    (storm forces)
FEAT_ENV    = 4   # wave_H, wind_V, water_depth, scour   (sea conditions)


# ============================================================================
# 1. DATA LOADER — reads 4 CSVs + ground truth
# ============================================================================
def load_csv_data():
    """Load data from separate CSV files and reshape into tensors.

    Returns dict of tensors, each (N_scenarios, N_steps, n_features).
    """
    # Check if CSV files exist; if not, generate them
    if not os.path.exists(os.path.join(DATA_DIR, "structural.csv")):
        print("CSV files not found — generating...")
        from generate_csv_data import generate
        generate()
        print()

    # Read CSVs
    df_struct = pd.read_csv(os.path.join(DATA_DIR, "structural.csv"))
    df_soil   = pd.read_csv(os.path.join(DATA_DIR, "soil.csv"))
    df_load   = pd.read_csv(os.path.join(DATA_DIR, "loading.csv"))
    df_env    = pd.read_csv(os.path.join(DATA_DIR, "environment.csv"))
    df_gt     = pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv"))

    n_scenarios = df_struct['scenario_id'].nunique()
    n_steps     = df_struct.groupby('scenario_id').size().iloc[0]

    print(f"Loaded {n_scenarios} scenarios × {n_steps} steps from CSV files.")

    def to_tensor(df, cols):
        """Reshape (N*T, C) → (N, T, C) tensor."""
        arr = df[cols].values.reshape(n_scenarios, n_steps, len(cols))
        return torch.tensor(arr, dtype=torch.float32)

    # Normalize each domain to [0, 1] range for stable training
    def normalize_df(df, cols):
        for c in cols:
            mn, mx = df[c].min(), df[c].max()
            if mx > mn:
                df[c] = (df[c] - mn) / (mx - mn)
            else:
                df[c] = 0.0
        return df

    struct_cols = ['L_pile_m', 'D_pile_m', 't_wall_m', 'EI_Nm2']
    soil_cols   = ['Su_kPa', 'G0_kPa', 'void_ratio', 'OCR']
    load_cols   = ['F_lateral_kN', 'M_base_kNm', 'N_cycles', 'freq_Hz']
    env_cols    = ['wave_H_m', 'wind_V_ms', 'water_depth_m', 'scour_depth_m']
    gt_cols     = ['u_displacement', 'K_stiffness', 'G_ratio', 'f_over_f0']

    # Store raw ground truth before normalization for metric computation
    gt_raw = to_tensor(df_gt, gt_cols)

    # Normalize features
    normalize_df(df_struct, struct_cols)
    normalize_df(df_soil, soil_cols)
    normalize_df(df_load, load_cols)
    normalize_df(df_env, env_cols)

    # Normalize ground truth targets too (store min/max for denorm)
    gt_stats = {}
    for c in gt_cols:
        mn, mx = df_gt[c].min(), df_gt[c].max()
        gt_stats[c] = (mn, mx)
        if mx > mn:
            df_gt[c] = (df_gt[c] - mn) / (mx - mn)

    return {
        'structural': to_tensor(df_struct, struct_cols),   # (N, T, 4)
        'soil':       to_tensor(df_soil, soil_cols),       # (N, T, 4)
        'loading':    to_tensor(df_load, load_cols),       # (N, T, 4)
        'environment':to_tensor(df_env, env_cols),         # (N, T, 4)
        'targets':    to_tensor(df_gt, gt_cols),           # (N, T, 4) normalized
        'targets_raw': gt_raw,                             # (N, T, 4) original scale
        'gt_stats':   gt_stats,
        'n_scenarios': n_scenarios,
        'n_steps':    n_steps,
    }


# ============================================================================
# 2. YIELD-SURFACE ACTIVATION  (the core novelty)
# ============================================================================
class YieldActivation(nn.Module):
    """Smooth activation analogous to a yield surface in engineering.

    Think of it like bending a metal rod:
      - Small forces: rod bends proportionally (linear → f(x) ≈ x)
      - Beyond yield: rod plastically deforms, response saturates

    Formula:  f(x) = x / (1 + |x / σ_y|^p )^(1/p)

    Parameters (ALL LEARNABLE — the model discovers them from data):
      σ_y : yield point per dimension — WHERE saturation begins
            (higher σ_y = the signal stays linear longer)
      p   : smoothness exponent — HOW SHARP the transition is
            (higher p = sharper elbow between linear and saturated)

    Physical interpretation per slot:
      Structural slot: σ_y ≈ pile elastic limit
      Soil slot:       σ_y ≈ undrained yield strain threshold
      Loading slot:    σ_y ≈ critical load level for nonlinearity
      Environment slot:σ_y ≈ threshold sea state for damage
      Fusion slot:     σ_y ≈ overall system yield point
    """
    def __init__(self, dim):
        super().__init__()
        self.sigma_y_raw = nn.Parameter(torch.randn(dim) * 0.1)
        self.p_raw       = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Ensure σ_y > 0 (softplus maps any number to a positive number)
        sigma_y = F.softplus(self.sigma_y_raw) + 0.01
        # Ensure p > 1.5 (controls sharpness of the yield "elbow")
        p       = 1.5 + F.softplus(self.p_raw)
        # Compute how far x is from the yield point
        ratio   = (x / sigma_y).abs().clamp(max=15.0)
        # Apply the yield formula: linear below σ_y, saturates above
        return x / (1.0 + ratio.pow(p)).pow(1.0 / p)

    def get_params(self):
        sigma_y = F.softplus(self.sigma_y_raw) + 0.01
        p       = 1.5 + F.softplus(self.p_raw)
        return sigma_y.detach(), p.detach()


# ============================================================================
# 3. SPECIALIZED SLOT — one per physical domain
# ============================================================================
class SpecializedSlot(nn.Module):
    """A single domain-specific slot (expert).

    Analogy: imagine 4 specialists examining a patient:
      - A cardiologist looks ONLY at heart data
      - A radiologist looks ONLY at X-ray images
      - etc.
    Each specialist forms their own assessment before consulting.

    Similarly, each slot:
      1. Embeds its domain features into a common 32-dim representation
      2. Uses self-attention to find patterns within its time series
      3. Applies yield activation to filter signals (domain-specific threshold)

    Crucially: each slot ONLY sees its own domain data.
    This enforces clean separation of physical concerns.
    """
    def __init__(self, n_features, name="slot"):
        super().__init__()
        self.name = name

        # Domain-specific embedding
        self.embed = nn.Sequential(
            nn.Linear(n_features, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

        # Lightweight self-attention to capture within-domain temporal patterns
        self.self_attn = nn.MultiheadAttention(
            D_MODEL, num_heads=2, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(D_MODEL)

        # Yield activation — THE core novelty per slot
        self.yield_act = YieldActivation(D_MODEL)

    def forward(self, x):
        """
        Args:  x: (B, T, n_features) — domain-specific features from CSV
        Returns: (B, T, D_MODEL) — slot representation after yield filtering
        """
        # Embed domain features
        h = self.embed(x)                             # (B, T, D)

        # Self-attention within the domain's time series
        h_n = self.norm(h)
        h_sa, _ = self.self_attn(h_n, h_n, h_n)      # (B, T, D)
        h = h + h_sa                                  # residual

        # Yield activation: filter signals that exceed the domain's yield
        h = self.yield_act(h)                         # (B, T, D)

        return h


# ============================================================================
# 4. FUSION SLOT — combines all specialized slots
# ============================================================================
class FusionSlot(nn.Module):
    """The 5th slot: combines all 4 specialized slot outputs.

    Analogy: after the 4 specialists each form their assessment,
    a lead doctor (the fusion slot) consults all of them to form
    the overall diagnosis.

    How it works:
      1. A learnable "fusion query" asks: "what’s most important?"
      2. Cross-attention lets the fusion query look at all 4 slots
         and assign an importance weight (α) to each one
      3. This is repeated 3 times (GRU refinement), getting sharper
         each round — like a doctor asking follow-up questions
      4. A yield activation filters the combined response

    The attention weights (α_0...α_3) tell us which domain matters
    most at each time step — providing interpretability.

    Physical meaning: the monopile's overall degradation depends
    on ALL domains together, not any single one.
    """
    def __init__(self):
        super().__init__()

        # Cross-attention: fusion slot queries from specialized slots
        self.cross_attn = nn.MultiheadAttention(
            D_MODEL, num_heads=2, batch_first=True, dropout=0.1)
        self.norm_q = nn.LayerNorm(D_MODEL)
        self.norm_kv = nn.LayerNorm(D_MODEL)

        # Learnable fusion query — initial state of the fusion slot
        self.fusion_query = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)

        # Iterative refinement (like slot attention — 3 rounds)
        self.gru = nn.GRUCell(D_MODEL, D_MODEL)

        # Yield activation for the combined system response
        self.yield_act = YieldActivation(D_MODEL)

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

    def forward(self, slot_outputs):
        """
        Args:
            slot_outputs: list of 4 tensors, each (B, T, D)
                          from the 4 specialized slots
        Returns:
            fused: (B, T, D) — combined representation for output heads
            attn_weights: (B, T, 4) — how much each domain contributed
        """
        B, T, D = slot_outputs[0].shape

        # Stack specialized outputs: (B, T, 4, D)
        stacked = torch.stack(slot_outputs, dim=2)  # (B, T, 4, D)

        # Reshape to treat each (batch, time) pair as a batch element
        # so cross-attention runs in ONE call (vectorized, no Python loop)
        kv_flat = stacked.reshape(B * T, 4, D)                # (B*T, 4, D)
        q_flat  = self.fusion_query.expand(B * T, -1, -1).clone()  # (B*T, 1, D)

        # Iterative refinement (3 rounds)
        for _ in range(SLOT_ITERS):
            q_n  = self.norm_q(q_flat)
            kv_n = self.norm_kv(kv_flat)
            out, aw = self.cross_attn(q_n, kv_n, kv_n)   # out: (B*T, 1, D)
            # GRU blends new info with previous state
            q_flat = self.gru(
                out.squeeze(1),       # (B*T, D)
                q_flat.squeeze(1)     # (B*T, D)
            ).unsqueeze(1)            # (B*T, 1, D)

        # Reshape back to (B, T, D)
        fused = q_flat.squeeze(1).reshape(B, T, D)
        attn_w = aw.squeeze(1).reshape(B, T, 4)  # last round's attention

        # Apply system-level yield activation
        fused = self.yield_act(fused)

        # Final projection
        fused = self.proj(fused)

        return fused, attn_w


# ============================================================================
# 5. FULL MODEL — Specialized Slots + Fusion + Encoder + Heads
# ============================================================================
def sinusoidal_pe(length, d):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    pe = torch.zeros(1, length, d)
    pos = torch.arange(length).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


class SpecializedSlotTransformer(nn.Module):
    """
    4 Specialized Slots (each reads its own CSV domain)
      + 1 Fusion Slot (combines everything)
      + Transformer Encoder
      + 4 Output Heads

    Design: deliberately SMALL ("less is better"):
      - d_model = 32
      - 1 encoder layer, 2 heads
      - 4+1 slots do the heavy lifting
    """
    def __init__(self, n_steps=30):
        super().__init__()

        # ── 4 Specialized Slots ──
        self.slot_structural = SpecializedSlot(FEAT_STRUCT, name="Structural")
        self.slot_soil       = SpecializedSlot(FEAT_SOIL,   name="Soil")
        self.slot_loading    = SpecializedSlot(FEAT_LOAD,   name="Loading")
        self.slot_environment= SpecializedSlot(FEAT_ENV,    name="Environment")

        # ── Fusion Slot (the 5th slot) ──
        self.fusion_slot = FusionSlot()

        # ── Positional encoding ──
        self.pe = nn.Parameter(sinusoidal_pe(n_steps, D_MODEL), requires_grad=False)

        # ── Transformer Encoder (1 layer, 2 heads — minimal) ──
        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=2, dim_feedforward=D_MODEL * 2,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # ── Slot aggregation: weighted sum of 4 specialized slots ──
        self.slot_gate = nn.Linear(D_MODEL * 4, D_MODEL)

        # ── 4 Output Heads (take fusion + slot aggregate) ──
        head_in = D_MODEL * 2  # fusion + slot aggregate
        self.head_u = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                    nn.Linear(D_MODEL, 1))                     # displacement
        self.head_K = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                    nn.Linear(D_MODEL, 1), nn.Sigmoid())       # stiffness ∈ [0,1]
        self.head_G = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                    nn.Linear(D_MODEL, 1), nn.Sigmoid())       # G/Gmax ∈ [0,1]
        self.head_f = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                    nn.Linear(D_MODEL, 1), nn.Sigmoid())       # f/f0 ∈ [0,1]

        # Storage for visualization
        self.last_fusion_attn = None
        self.last_slot_outputs = None

    def forward(self, x_struct, x_soil, x_load, x_env):
        """
        Args:
            x_struct: (B, T, 4) from structural.csv
            x_soil:   (B, T, 4) from soil.csv
            x_load:   (B, T, 4) from loading.csv
            x_env:    (B, T, 4) from environment.csv
        Returns:
            u, K, G, f — each (B, T)
        """
        B, T, _ = x_struct.shape

        # ── Step 1: Each specialized slot processes its own domain ──
        h_struct = self.slot_structural(x_struct)    # (B, T, D)
        h_soil   = self.slot_soil(x_soil)            # (B, T, D)
        h_load   = self.slot_loading(x_load)         # (B, T, D)
        h_env    = self.slot_environment(x_env)      # (B, T, D)

        slot_outputs = [h_struct, h_soil, h_load, h_env]
        self.last_slot_outputs = [h.detach() for h in slot_outputs]

        # ── Step 2: Fusion slot combines all domains ──
        fused, fusion_attn = self.fusion_slot(slot_outputs)  # (B, T, D)
        self.last_fusion_attn = fusion_attn.detach()         # (B, T, 4)

        # ── Step 3: Aggregate specialized slot outputs (direct pathway) ──
        slot_cat = torch.cat(slot_outputs, dim=-1)           # (B, T, 4*D)
        slot_agg = self.slot_gate(slot_cat)                  # (B, T, D)

        # ── Step 4: Add positional encoding + transformer encoder ──
        x = fused + self.pe[:, :T]
        x = self.encoder(x)                          # (B, T, D)

        # ── Step 5: Concat fusion output + slot aggregate for heads ──
        x_out = torch.cat([x, slot_agg], dim=-1)    # (B, T, 2*D)

        # ── Step 6: Output heads ──
        u = self.head_u(x_out).squeeze(-1)           # (B, T)
        K = self.head_K(x_out).squeeze(-1)           # (B, T)
        G = self.head_G(x_out).squeeze(-1)           # (B, T)
        f = self.head_f(x_out).squeeze(-1)           # (B, T)

        return u, K, G, f


# ============================================================================
# 6. LOSS FUNCTION  (physics-informed, 4 outputs)
# ============================================================================
def loss_fn(u_p, K_p, G_p, f_p, targets):
    """Physics-informed loss function.

    Instead of only measuring how close predictions are to truth,
    we also enforce physical rules:

    Terms:
      L_data    — Mean Squared Error for all 4 outputs (accuracy)
      L_smooth  — Penalize jerky displacement predictions (smooth u)
      L_mono_G  — G/Gmax should decrease over time   (soil degrades)
      L_mono_f  — f/f0 should decrease                (damage grows)
      L_mono_u  — u should increase                   (pile accumulates displacement)

    The physics penalties (smooth + monotonicity) act as "guardrails"
    ensuring the model respects known physical behavior even when
    the data alone might not enforce it perfectly.
    """
    u_t = targets[:, :, 0]
    K_t = targets[:, :, 1]
    G_t = targets[:, :, 2]
    f_t = targets[:, :, 3]

    # Data fitting
    L_u = F.mse_loss(u_p, u_t)
    L_K = F.mse_loss(K_p, K_t)
    L_G = F.mse_loss(G_p, G_t)
    L_f = F.mse_loss(f_p, f_t)

    # Smoothness: penalize large 2nd derivative of u
    du  = u_p[:, 1:] - u_p[:, :-1]
    d2u = du[:, 1:] - du[:, :-1]
    L_smooth = d2u.pow(2).mean()

    # Monotonicity: G and f decrease; u increases
    L_mono = (F.relu(G_p[:, 1:] - G_p[:, :-1]).mean()
              + F.relu(f_p[:, 1:] - f_p[:, :-1]).mean()
              + F.relu(-(u_p[:, 1:] - u_p[:, :-1])).mean())

    total = L_u + L_K + L_G + L_f + 0.2 * L_smooth + 0.15 * L_mono

    return total, dict(total=total, u=L_u, K=L_K, G=L_G, f=L_f, smooth=L_smooth)


# ============================================================================
# 7. METRICS
# ============================================================================
def r2(p, t):
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

def mape(p, t):
    return ((t - p).abs() / t.clamp(min=0.01)).mean().item() * 100


# ============================================================================
# 8. TRAINING LOOP
# ============================================================================
def train():
    sep = "=" * 60
    print(f"\n{sep}")
    print("Meeting 2 — Specialized-Slot Yield Transformer  (v2)")
    print(f"{sep}")
    print("Each slot handles ONE physical domain:")
    print("  Slot 0: Structural  (pile geometry & material)")
    print("  Slot 1: Soil        (strength, stiffness, state)")
    print("  Slot 2: Loading     (forces, cycles, frequency)")
    print("  Slot 3: Environment (waves, wind, scour)")
    print("  Slot 4: FUSION      (combines all → degradation)")
    print()

    # ── Load CSV data ──
    data = load_csv_data()
    N = data['n_scenarios']
    T = data['n_steps']
    n_tr = int(0.85 * N)
    tr = torch.arange(n_tr)
    te = torch.arange(n_tr, N)

    # ── Build model ──
    model = SpecializedSlotTransformer(n_steps=T)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:  {n_params:,}  (target: as few as possible)")
    print(f"Specialized slots: {N_SPEC_SLOTS},  + 1 Fusion slot,  d_model: {D_MODEL}")
    print(f"Training {EPOCHS} epochs  ({n_tr} train / {len(te)} test)\n")

    # ── Train ──
    keys = ['total', 'u', 'K', 'G', 'f', 'smooth']
    hist = {k: [] for k in keys}

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        sums = {k: 0.0 for k in keys}
        nb = 0

        for i in range(0, n_tr, BATCH):
            idx = perm[i:i + BATCH]

            u_p, K_p, G_p, f_p = model(
                data['structural'][idx],
                data['soil'][idx],
                data['loading'][idx],
                data['environment'][idx],
            )

            loss, losses = loss_fn(u_p, K_p, G_p, f_p, data['targets'][idx])

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k in keys:
                sums[k] += losses[k].item()
            nb += 1

        for k in keys:
            hist[k].append(sums[k] / nb)
        sched.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            h = hist
            print(f"  Ep {ep+1:3d}/{EPOCHS}  "
                  f"Tot:{h['total'][-1]:.4f}  u:{h['u'][-1]:.4f}  "
                  f"K:{h['K'][-1]:.4f}  G:{h['G'][-1]:.4f}  "
                  f"f:{h['f'][-1]:.4f}  sm:{h['smooth'][-1]:.6f}")

    # ── Test ──
    print(f"\n{'─' * 40}\nTest Results\n{'─' * 40}")
    model.eval()
    with torch.no_grad():
        u_p, K_p, G_p, f_p = model(
            data['structural'][te],
            data['soil'][te],
            data['loading'][te],
            data['environment'][te],
        )
        tgt = data['targets'][te]
        u_t, K_t, G_t, f_t = tgt[:,:,0], tgt[:,:,1], tgt[:,:,2], tgt[:,:,3]

        for nm, pred, true in [("u (disp)", u_p, u_t),
                                ("K (stiff)", K_p, K_t),
                                ("G/Gmax", G_p, G_t),
                                ("f/f0", f_p, f_t)]:
            print(f"  {nm:12s}  MSE={F.mse_loss(pred,true):.6f}  "
                  f"R²={r2(pred,true):.4f}  MAPE={mape(pred,true):.2f}%")

    # ── Show self-calibrated yield parameters ──
    print(f"\nSelf-calibrated yield parameters:")
    slot_names = ["Structural", "Soil", "Loading", "Environment", "Fusion"]
    slots = [model.slot_structural, model.slot_soil,
             model.slot_loading, model.slot_environment]
    for i, (slot, name) in enumerate(zip(slots, slot_names[:4])):
        sy, pp = slot.yield_act.get_params()
        print(f"  Slot {i} ({name:12s}): σ_y={sy.mean():.3f}  p={pp.item():.2f}")
    # Fusion slot
    sy, pp = model.fusion_slot.yield_act.get_params()
    print(f"  Slot 4 ({'Fusion':12s}): σ_y={sy.mean():.3f}  p={pp.item():.2f}")

    # ── Visualize ──
    visualize(model, data, te, hist)
    out_path = os.path.join(SCRIPT_DIR, 'specialized_slot_results.png')
    print(f"\n{sep}\nDone. Saved → {out_path}\n{sep}")
    return model


# ============================================================================
# 9. VISUALIZATION  (10-panel figure)
# ============================================================================
def visualize(model, data, te, hist):
    """Generate 2×5 = 10-panel figure.

    Row 1: (a) Losses, (b) Load-Disp, (c) G/Gmax, (d) Stiffness, (e) f/f0
    Row 2: (f) Yield shapes, (g) Fusion attn, (h) Scatter u, (i) Scatter f/f0, (j) Domain contrib
    """
    fig, ax = plt.subplots(2, 5, figsize=(28, 10))
    fig.suptitle("Meeting 2 — Specialized-Slot Yield Transformer: Domain-Specific + Fusion",
                 fontsize=14, fontweight='bold')

    model.eval()
    with torch.no_grad():
        u_p, K_p, G_p, f_p = model(
            data['structural'][te],
            data['soil'][te],
            data['loading'][te],
            data['environment'][te],
        )
    tgt = data['targets'][te]
    u_t, K_t, G_t, f_t = tgt[:,:,0], tgt[:,:,1], tgt[:,:,2], tgt[:,:,3]
    ns = min(4, len(te))
    t = np.arange(data['n_steps'])

    # ── (a) Training losses ──
    for k, ls in [('total','-'),('u','--'),('K','-.'),('G',':'),('f','--'),('smooth',':')]:
        ax[0,0].plot(hist[k], label=k, ls=ls)
    ax[0,0].set(title='(a) Training Losses', xlabel='Epoch', ylabel='Loss', yscale='log')
    ax[0,0].legend(fontsize=5)
    ax[0,0].grid(True, alpha=0.3)

    # ── (b) Load-displacement ──
    F_load = data['loading'][te][:, :, 0]  # F_lateral (normalized)
    for i in range(ns):
        ax[0,1].plot(F_load[i].numpy(), u_t[i].numpy(), 'o-', ms=2, alpha=0.5, color=f'C{i}')
        ax[0,1].plot(F_load[i].numpy(), u_p[i].numpy(), 's--', ms=2, alpha=0.7, color=f'C{i}')
    ax[0,1].set(title='(b) Load–Disp (o=true, s=pred)', xlabel='F (norm)', ylabel='u (norm)')
    ax[0,1].grid(True, alpha=0.3)

    # ── (c) G/Gmax over time ──
    for i in range(ns):
        ax[0,2].plot(t, G_t[i].numpy(), 'o-', ms=2, alpha=0.5, color=f'C{i}')
        ax[0,2].plot(t, G_p[i].numpy(), 's--', ms=2, alpha=0.7, color=f'C{i}')
    ax[0,2].set(title='(c) G/Gmax (soil degradation)', xlabel='Step')
    ax[0,2].set_ylim(-0.05, 1.1)
    ax[0,2].grid(True, alpha=0.3)

    # ── (d) Stiffness over time ──
    for i in range(ns):
        ax[0,3].plot(t, K_t[i].numpy(), 'o-', ms=2, alpha=0.5, color=f'C{i}')
        ax[0,3].plot(t, K_p[i].numpy(), 's--', ms=2, alpha=0.7, color=f'C{i}')
    ax[0,3].set(title='(d) Stiffness K', xlabel='Step')
    ax[0,3].grid(True, alpha=0.3)

    # ── (e) Natural frequency ratio f/f0 ──
    for i in range(ns):
        ax[0,4].plot(t, f_t[i].numpy(), 'o-', ms=2, alpha=0.5, color=f'C{i}')
        ax[0,4].plot(t, f_p[i].numpy(), 's--', ms=2, alpha=0.7, color=f'C{i}')
    ax[0,4].set(title='(e) Freq ratio f/f₀', xlabel='Step')
    ax[0,4].set_ylim(0.2, 1.1)
    ax[0,4].grid(True, alpha=0.3)

    # ── (f) Learned yield activation shapes (all 5 slots) ──
    slot_names = ["Structural", "Soil", "Loading", "Environment", "Fusion"]
    slots_ya = [model.slot_structural.yield_act, model.slot_soil.yield_act,
                model.slot_loading.yield_act, model.slot_environment.yield_act,
                model.fusion_slot.yield_act]
    x_range = torch.linspace(-2, 2, 300)
    colors = ['#2196F3', '#8D6E63', '#FF5722', '#00BCD4', '#9C27B0']
    for idx, (ya, nm, col) in enumerate(zip(slots_ya, slot_names, colors)):
        sy, pp = ya.get_params()
        sy_m = sy.mean()
        p_v = pp.item() if pp.dim() == 0 else pp.mean().item()
        ratio = (x_range / sy_m).abs().clamp(max=15.0)
        y = x_range / (1.0 + ratio.pow(p_v)).pow(1.0 / p_v)
        ls = '-' if idx < 4 else '--'
        lw = 1.5 if idx < 4 else 2.5
        ax[1,0].plot(x_range.numpy(), y.numpy(), color=col, ls=ls, lw=lw,
                     label=f'{nm}: σ_y={sy_m:.2f}, p={p_v:.1f}')
    ax[1,0].axhline(0, color='k', lw=0.5); ax[1,0].axvline(0, color='k', lw=0.5)
    ax[1,0].set(title='(f) Yield Activations (all 5 slots)', xlabel='x', ylabel='f(x)')
    ax[1,0].legend(fontsize=5)
    ax[1,0].grid(True, alpha=0.3)

    # ── (g) Fusion attention heatmap (which domain contributes when) ──
    if model.last_fusion_attn is not None:
        fa = model.last_fusion_attn[0].numpy()  # (T, 4)
        im = ax[1,1].imshow(fa.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax[1,1].set_yticks(range(4))
        ax[1,1].set_yticklabels(['Struct', 'Soil', 'Load', 'Env'], fontsize=7)
        plt.colorbar(im, ax=ax[1,1], fraction=0.046)
    ax[1,1].set(title='(g) Fusion Attention Map', xlabel='Time step', ylabel='Domain')

    # ── (h) Scatter: u predicted vs true ──
    ax[1,2].scatter(u_t.flatten().numpy(), u_p.flatten().numpy(),
                    s=1, alpha=0.3, c='#1976D2')
    lim = max(u_t.max().item(), u_p.max().item()) * 1.05
    ax[1,2].plot([0, lim], [0, lim], 'k--', lw=1)
    ax[1,2].set(title='(h) u: Pred vs True', xlabel='True', ylabel='Predicted')
    ax[1,2].set_aspect('equal')
    ax[1,2].grid(True, alpha=0.3)

    # ── (i) Scatter: f/f0 predicted vs true ──
    ax[1,3].scatter(f_t.flatten().numpy(), f_p.flatten().numpy(),
                    s=1, alpha=0.3, c='#9C27B0')
    lim_f = max(f_t.max().item(), f_p.max().item()) * 1.05
    ax[1,3].plot([0, lim_f], [0, lim_f], 'k--', lw=1)
    ax[1,3].set(title='(i) f/f₀: Pred vs True', xlabel='True', ylabel='Predicted')
    ax[1,3].set_aspect('equal')
    ax[1,3].grid(True, alpha=0.3)

    # ── (j) Domain contribution over time (stacked area) ──
    if model.last_fusion_attn is not None:
        fa_mean = model.last_fusion_attn.mean(dim=0).numpy()  # (T, 4) averaged over batch
        ax[1,4].stackplot(t, fa_mean.T,
                          labels=['Structural', 'Soil', 'Loading', 'Environ.'],
                          colors=['#2196F3', '#8D6E63', '#FF5722', '#00BCD4'], alpha=0.8)
        ax[1,4].set(title='(j) Domain Contribution Over Time',
                    xlabel='Step', ylabel='Attention weight')
        ax[1,4].legend(fontsize=5, loc='lower right')
        ax[1,4].set_ylim(0, 1.05)
    ax[1,4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'specialized_slot_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
if __name__ == '__main__':
    train()
