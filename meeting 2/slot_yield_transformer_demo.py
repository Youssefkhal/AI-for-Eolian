"""
=============================================================================
Meeting 2 — Slot-Yield Transformer Demo
=============================================================================
Professor's notes:
  - Slots with activation functions ANALOGOUS TO YIELD SURFACES
  - Function to SMOOTHEN load-displacement curve
  - SELF-CALIBRATING governing parameters
  - LESS parameters = BETTER

Key idea:
  Each SLOT has a learnable yield-surface activation function:

      f(x) = x / (1 + |x / σ_y|^p )^(1/p)

  Below yield (|x| << σ_y):  f(x) ≈ x          (elastic / linear)
  Above yield (|x| >> σ_y):  f(x) → ±σ_y        (plastic / saturated)

  σ_y (yield point) and p (smoothness) are nn.Parameters — LEARNED
  from data, not hand-tuned.  This is the "self-calibrating" part.

  Multiple slots = multiple yield surfaces, each capturing a different
  behavioral regime of the soil (elastic, transition, plastic, post-peak).

Architecture:
  Input  [F_t, u_{t-1}, K_{t-1}, L, D]  →  5 values per step
  ─────────────────────────────────────────────────────────────
  Embedding (5 → 32)
       ↓
  Slot Attention  (4 slots × 3 iterations, yield activations)
       ↓
  Transformer Encoder  (1 layer, 2 heads — "less is better")
       ↓
  Output Heads → u_t (displacement), K_t (stiffness), G/Gmax_t (soil health)
  ─────────────────────────────────────────────────────────────
  Output  [u_t, K_t, G/Gmax_t]  →  3 values per step

Physics:
  Load-displacement follows a smooth yield model:
    u(F) = (F / K₀) · (1 + (F/F_y)^n)^(1/n)
  G/Gmax follows Hardin-Drnevich style degradation.
  The yield activation is the INVERSE of this curve — so the model
  self-calibrates σ_y ≈ F_y and p ≈ n from data.

Sources:
  - Locatello et al. (2020), Object-Centric Learning with Slot Attention
  - Vaswani et al. (2017), Attention Is All You Need
  - Hardin & Drnevich (1972), shear modulus degradation
  - Vucetic & Dobry (1991), G/Gmax curves
  - Bhattacharya (2019), OWT natural frequency
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ──────────────────────────────────────────────────────────────────
N_SAMPLES = 200          # synthetic scenarios
SEQ_LEN   = 30           # loading steps per scenario
N_SLOTS   = 4            # number of behavioral-regime slots
D_MODEL   = 32           # transformer hidden dim ("less is better")
EPOCHS    = 100          # training epochs
BATCH     = 32           # mini-batch size
LR        = 2e-3         # learning rate


# ============================================================================
# 1. YIELD-SURFACE ACTIVATION  (the core novelty)
# ============================================================================
class YieldActivation(nn.Module):
    """Smooth activation analogous to a yield surface.

    f(x) = x / (1 + |x / σ_y|^p )^(1/p)

    Parameters (ALL LEARNABLE — self-calibrating):
      σ_y  : yield point per slot per dimension.
              Below σ_y the response is linear (elastic).
              Above σ_y the response saturates (plastic).
      p    : smoothness exponent per slot.
              Higher p → sharper elbow at yield.
              Lower p  → smoother, more gradual transition.

    The model LEARNS σ_y and p from data via backpropagation.
    This is the "self-calibrating governing parameters" the professor described.
    """
    def __init__(self, n_slots, dim):
        super().__init__()
        # Raw parameters (will be passed through softplus to ensure positivity)
        self.sigma_y_raw = nn.Parameter(torch.randn(n_slots, dim) * 0.1)
        self.p_raw = nn.Parameter(torch.zeros(n_slots))

    def forward(self, x, slot_idx):
        """Apply yield activation for a specific slot.

        Args:
            x         : (B, dim) — values coming into this slot
            slot_idx  : int — which slot's yield parameters to use
        Returns:
            Activated values, same shape as x.
        """
        # σ_y > 0 always (softplus + small offset for stability)
        sigma_y = F.softplus(self.sigma_y_raw[slot_idx]) + 0.01
        # p > 1.5 always (ensures smooth, differentiable transition)
        p = 1.5 + F.softplus(self.p_raw[slot_idx])

        # Core yield formula (clamped to prevent overflow in pow)
        ratio = (x / sigma_y).abs().clamp(max=15.0)
        return x / (1.0 + ratio.pow(p)).pow(1.0 / p)

    def get_params(self):
        """Return the current (learned) yield parameters for inspection."""
        sigma_y = F.softplus(self.sigma_y_raw) + 0.01
        p = 1.5 + F.softplus(self.p_raw)
        return sigma_y.detach(), p.detach()


# ============================================================================
# 2. SLOT ATTENTION MODULE  (with yield activations)
# ============================================================================
class SlotAttention(nn.Module):
    """Slot Attention (Locatello et al. 2020) extended with yield activations.

    K learnable slots compete for input positions via iterative cross-attention.
    After each attention step, the update to each slot passes through that
    slot's YIELD ACTIVATION — filtering out signals above yield (saturation).

    This means each slot naturally specializes in a different BEHAVIORAL REGIME:
      - Slot with low σ_y  → captures small-strain elastic behavior
      - Slot with high σ_y → captures large-strain plastic behavior
    """
    def __init__(self, dim, n_slots, n_iters=3):
        super().__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters

        # Learnable slot centers (initial positions in feature space)
        self.slot_mu = nn.Parameter(torch.randn(n_slots, dim) * (dim ** -0.5))

        # Attention projections (query from slots, key/value from input)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # GRU for iterative slot refinement (recurrent update)
        self.gru = nn.GRUCell(dim, dim)

        # Layer norms
        self.norm_s = nn.LayerNorm(dim)
        self.norm_x = nn.LayerNorm(dim)

        # Yield activation — one per slot
        self.yield_act = YieldActivation(n_slots, dim)

    def forward(self, x):
        """
        Args:
            x : (B, T, D) — embedded input features
        Returns:
            slots     : (B, K, D) — final slot representations
            final_attn: (B, K, T) — attention weights (for visualization)
        """
        B, T, D = x.shape
        K = self.n_slots

        # Initialize all slots from learned centers
        slots = self.slot_mu.unsqueeze(0).expand(B, -1, -1).clone()  # (B, K, D)

        # Pre-compute keys and values from input (constant across iterations)
        x_n = self.norm_x(x)
        k = self.to_k(x_n)   # (B, T, D)
        v = self.to_v(x_n)   # (B, T, D)

        # Iterative refinement (3 rounds by default)
        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.to_q(self.norm_s(slots))   # (B, K, D)

            # Attention: softmax over SLOTS (slots compete for each position)
            dots = torch.einsum('bkd,btd->bkt', q, k) * (D ** -0.5)
            attn = F.softmax(dots, dim=1)           # (B, K, T) — sum over K = 1

            # Weighted mean of values for each slot
            attn_n = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum('bkt,btd->bkd', attn_n, v)  # (B, K, D)

            # ── KEY STEP: apply yield activation to each slot's update ──
            # (out-of-place to keep autograd happy)
            activated = [self.yield_act(updates[:, s], slot_idx=s) for s in range(K)]
            updates = torch.stack(activated, dim=1)  # (B, K, D)

            # GRU update: blend new information with previous slot state
            slots = self.gru(
                updates.reshape(B * K, D),
                slots_prev.reshape(B * K, D)
            ).reshape(B, K, D)

        # Compute final attention weights (for XAI visualization)
        q = self.to_q(self.norm_s(slots))
        dots = torch.einsum('bkd,btd->bkt', q, k) * (D ** -0.5)
        final_attn = F.softmax(dots, dim=1)

        return slots, final_attn


# ============================================================================
# 3. SLOT-YIELD TRANSFORMER MODEL
# ============================================================================
def sinusoidal_pe(length, d):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    pe = torch.zeros(1, length, d)
    pos = torch.arange(length).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


class SlotYieldTransformer(nn.Module):
    """Complete model: Embedding → SlotAttention → Encoder → Outputs.

    Input  per step: [F_t, u_{t-1}, K_{t-1}, L, D]  = 5 values
    Output per step: [u_t, K_t, G/Gmax_t]            = 3 values

    Design: deliberately SMALL ("less is better"):
      - d_model = 32  (vs 64+ in typical transformers)
      - 1 encoder layer (vs 2-6 typical)
      - 2 attention heads (vs 4-8 typical)
      - 4 slots do the heavy lifting
    """
    def __init__(self):
        super().__init__()
        # Embedding: 5 raw inputs → d_model
        self.embed = nn.Sequential(
            nn.Linear(5, D_MODEL), nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        # Positional encoding (fixed, not learned)
        self.pe = nn.Parameter(sinusoidal_pe(SEQ_LEN, D_MODEL), requires_grad=False)

        # Slot attention with yield activations (the core innovation)
        self.slot_attn = SlotAttention(D_MODEL, N_SLOTS, n_iters=3)

        # Lightweight transformer encoder (1 layer, 2 heads)
        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=2, dim_feedforward=D_MODEL * 2,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Output heads
        self.head_u = nn.Linear(D_MODEL, 1)                              # displacement
        self.head_K = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Softplus())  # stiffness > 0
        self.head_G = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Sigmoid())   # G/Gmax ∈ [0,1]

        self.last_attn = None  # stored for visualization

    def forward(self, F_t, u_prev, K_prev, L, D):
        """
        Args:
            F_t    : (B, T) — applied force at each step
            u_prev : (B, T) — displacement at previous step (autoregressive)
            K_prev : (B, T) — stiffness at previous step (autoregressive)
            L      : (B, 1) — pile length (static)
            D      : (B, 1) — pile diameter (static)
        Returns:
            u_pred, K_pred, G_pred — each (B, T)
        """
        B, T = F_t.shape

        # Stack all inputs: (B, T, 5)
        inp = torch.stack([
            F_t,                       # current force
            u_prev,                    # previous displacement
            K_prev,                    # previous stiffness
            L.expand(-1, T),           # pile length (broadcast to all steps)
            D.expand(-1, T),           # pile diameter (broadcast to all steps)
        ], dim=-1)

        # Embed + positional encoding
        x = self.embed(inp) + self.pe[:, :T]       # (B, T, 32)

        # Slot attention: slots learn yield-surface regimes
        slots, attn = self.slot_attn(x)             # slots (B,4,32), attn (B,4,T)
        self.last_attn = attn.detach()

        # Broadcast slots back to sequence (each position gets its slot mixture)
        slot_ctx = torch.einsum('bkt,bkd->btd', attn, slots)  # (B, T, 32)
        x = x + slot_ctx                                       # residual connection

        # Transformer encoder for temporal dependencies
        x = self.encoder(x)                         # (B, T, 32)

        # Output heads
        u = self.head_u(x).squeeze(-1)              # (B, T)
        K = self.head_K(x).squeeze(-1)              # (B, T) always > 0
        G = self.head_G(x).squeeze(-1)              # (B, T) always in [0, 1]
        return u, K, G


# ============================================================================
# 4. SYNTHETIC DATA  (load-displacement with yield behavior)
# ============================================================================
def make_data():
    """Generate load-displacement data following a smooth yield model.

    Physics:
      u(F) = (F / K₀) · (1 + (F/F_y)^n)^(1/n)

      - F << F_y:  u ≈ F / K₀           (linear / elastic)
      - F >> F_y:  u ≈ F² / (K₀ · F_y)  (nonlinear / yielding)
      - F_y controls WHERE yield begins
      - n  controls HOW SHARP the transition is

    G/Gmax:
      G/Gmax = 1 / (1 + (strain / strain_ref)^1.2)
      (Hardin-Drnevich style degradation)
    """
    N, T = N_SAMPLES, SEQ_LEN

    # ── Pile geometry (constant per scenario) ──
    L = torch.rand(N, 1) * 0.8 + 0.2       # pile length   [0.2, 1.0]
    D = torch.rand(N, 1) * 0.6 + 0.2       # pile diameter [0.2, 0.8]

    # ── Soil yield properties (GROUND TRUTH the model must self-calibrate to) ──
    K0  = 2.0 * D + 1.5 * L + 0.5 + 0.1 * torch.randn(N, 1).abs()  # initial stiffness
    F_y = 0.5 * D + 0.3 * L + 0.2 + 0.05 * torch.randn(N, 1).abs() # yield force
    n_s = 2.0 + 0.5 * torch.rand(N, 1)                               # smoothness exponent

    # ── Monotonically increasing force ──
    F_max  = F_y * 2.5
    t_norm = torch.linspace(0, 1, T).unsqueeze(0).expand(N, -1)
    Ft     = (F_max * t_norm + 0.005 * torch.randn(N, T)).clamp(min=0)

    # ── Displacement from smooth yield model ──
    ratio = (Ft / F_y.expand(-1, T)).clamp(min=0)
    u = (Ft / K0.expand(-1, T)) * (1.0 + ratio.pow(n_s.expand(-1, T))).pow(
        1.0 / n_s.expand(-1, T))
    u = u + 0.003 * torch.randn_like(u).abs()  # small measurement noise

    # ── G/Gmax degradation (Hardin-Drnevich style) ──
    strain = u / (L.expand(-1, T) * 0.5)
    G = 1.0 / (1.0 + strain.clamp(min=0).pow(1.2))

    # ── Tangent stiffness ──
    K = K0.expand(-1, T) * G

    # ── Autoregressive: shift right for previous-step inputs ──
    u_prev = torch.cat([torch.zeros(N, 1), u[:, :-1]], dim=1)
    K_prev = torch.cat([K0, K[:, :-1]], dim=1)

    return dict(
        L=L, D=D, F=Ft,
        u_true=u, G_true=G, K_true=K,
        u_prev=u_prev, K_prev=K_prev,
        K0=K0, F_y=F_y, n_smooth=n_s,
    )


# ============================================================================
# 5. LOSS FUNCTION  (physics-informed)
# ============================================================================
def loss_fn(u_p, K_p, G_p, u_t, K_t, G_t):
    """Physics-informed loss with smoothness and monotonicity penalties.

    Terms:
      1. L_u      — displacement MSE (data fitting)
      2. L_K      — stiffness MSE (data fitting)
      3. L_G      — G/Gmax MSE (data fitting)
      4. L_smooth — penalize jerky predictions (2nd derivative of u)
      5. L_mono   — G/Gmax must decrease; u must increase over time
      6. L_pos    — stiffness > 0, G/Gmax ∈ [0, 1]
    """
    # Data fitting
    L_u = F.mse_loss(u_p, u_t)
    L_K = F.mse_loss(K_p, K_t)
    L_G = F.mse_loss(G_p, G_t)

    # Smoothness: penalize large 2nd derivative (jerky curves)
    du = u_p[:, 1:] - u_p[:, :-1]
    d2u = du[:, 1:] - du[:, :-1]
    L_smooth = d2u.pow(2).mean()

    # Monotonicity: displacement up, G/Gmax down
    L_mono = (F.relu(G_p[:, 1:] - G_p[:, :-1]).mean()
              + F.relu(-(u_p[:, 1:] - u_p[:, :-1])).mean())

    # Bounds: K > 0, G ∈ [0, 1]
    L_pos = F.relu(-K_p).mean() + F.relu(-G_p).mean() + F.relu(G_p - 1).mean()

    total = L_u + L_K + L_G + 0.3 * L_smooth + 0.2 * L_mono + 0.2 * L_pos
    return total, dict(total=total, u=L_u, K=L_K, G=L_G, smooth=L_smooth)


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
def r2(p, t):
    """R² (coefficient of determination)."""
    return (1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()

def mape(p, t):
    """Mean Absolute Percentage Error (%)."""
    return ((t - p).abs() / t.clamp(min=0.01)).mean().item() * 100


def train():
    sep = "=" * 60
    print(f"{sep}\nMeeting 2 — Slot-Yield Transformer Demo\n{sep}")
    print("Concept: Slots with yield-surface activations,")
    print("         self-calibrating parameters, less = better.\n")

    # ── Generate data ──
    data = make_data()
    n_tr = int(0.85 * N_SAMPLES)
    tr = torch.arange(n_tr)
    te = torch.arange(n_tr, N_SAMPLES)

    # ── Build model ──
    model = SlotYieldTransformer()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters:  {n_params:,}  (target: as few as possible)")
    print(f"Slots: {N_SLOTS},  d_model: {D_MODEL},  encoder layers: 1")
    print(f"Training {EPOCHS} epochs  ({n_tr} train / {len(te)} test)\n")

    # ── Train ──
    keys = ['total', 'u', 'K', 'G', 'smooth']
    hist = {k: [] for k in keys}

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        sums = {k: 0.0 for k in keys}
        nb = 0

        for i in range(0, n_tr, BATCH):
            idx = perm[i:i + BATCH]
            u_p, K_p, G_p = model(
                data['F'][idx], data['u_prev'][idx], data['K_prev'][idx],
                data['L'][idx], data['D'][idx])
            loss, losses = loss_fn(
                u_p, K_p, G_p,
                data['u_true'][idx], data['K_true'][idx], data['G_true'][idx])

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
                  f"smooth:{h['smooth'][-1]:.6f}")

    # ── Test ──
    print("\n--- Test ---")
    model.eval()
    with torch.no_grad():
        u_p, K_p, G_p = model(
            data['F'][te], data['u_prev'][te], data['K_prev'][te],
            data['L'][te], data['D'][te])
        u_t = data['u_true'][te]
        K_t = data['K_true'][te]
        G_t = data['G_true'][te]

        for nm, p, t in [("u", u_p, u_t), ("K", K_p, K_t), ("G/Gmax", G_p, G_t)]:
            print(f"  {nm:7s}  MSE={F.mse_loss(p,t):.6f}  "
                  f"R²={r2(p,t):.4f}  MAPE={mape(p,t):.2f}%")

    # ── Show self-calibrated yield parameters ──
    sy, pp = model.slot_attn.yield_act.get_params()
    print(f"\nSelf-calibrated yield parameters (learned from data):")
    for s in range(N_SLOTS):
        print(f"  Slot {s}: σ_y = {sy[s].mean():.3f}   p = {pp[s]:.2f}")
    print(f"  (Compare: true F_y range = "
          f"[{data['F_y'].min():.3f}, {data['F_y'].max():.3f}])")
    print(f"  (Compare: true n  range  = "
          f"[{data['n_smooth'].min():.2f}, {data['n_smooth'].max():.2f}])")

    # ── Visualize ──
    visualize(model, data, te, hist)
    out_path = os.path.join(SCRIPT_DIR, 'slot_yield_results.png')
    print(f"\n{sep}\nDone. Saved → {out_path}\n{sep}")
    return model


# ============================================================================
# 7. VISUALIZATION  (8-panel figure)
# ============================================================================
def visualize(model, data, te, hist):
    """Generate 2×4 = 8-panel figure.

    Row 1: (a) Loss, (b) Load-displacement, (c) G/Gmax, (d) Stiffness
    Row 2: (e) Yield activations, (f) Slot heatmap, (g) u scatter, (h) Smoothness
    """
    fig, ax = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Meeting 2 — Slot-Yield Transformer: Self-Calibrating Yield Surfaces",
                 fontsize=13)

    model.eval()
    with torch.no_grad():
        u_p, K_p, G_p = model(
            data['F'][te], data['u_prev'][te], data['K_prev'][te],
            data['L'][te], data['D'][te])
    u_t = data['u_true'][te]
    K_t = data['K_true'][te]
    G_t = data['G_true'][te]
    ns = min(4, len(te))
    t = np.arange(SEQ_LEN)

    # ── (a) Training losses ──
    for k, ls in [('total', '-'), ('u', '--'), ('K', '-.'), ('G', ':'), ('smooth', ':')]:
        ax[0, 0].plot(hist[k], label=k, ls=ls)
    ax[0, 0].set(title='(a) Training Losses', xlabel='Epoch', ylabel='Loss', yscale='log')
    ax[0, 0].legend(fontsize=6)
    ax[0, 0].grid(True, alpha=0.3)

    # ── (b) Load-displacement curves ──
    for i in range(ns):
        F_i = data['F'][te[i]].numpy()
        ax[0, 1].plot(F_i, u_t[i].numpy(), 'o-', ms=2, alpha=0.5, color=f'C{i}')
        ax[0, 1].plot(F_i, u_p[i].numpy(), 's--', ms=2, alpha=0.7, color=f'C{i}')
    ax[0, 1].set(title='(b) Load–Displacement  (o=true, s=pred)',
                 xlabel='Force $F$', ylabel='Displacement $u$')
    ax[0, 1].grid(True, alpha=0.3)

    # ── (c) G/Gmax over time ──
    for i in range(ns):
        ax[0, 2].plot(t, G_t[i].numpy(), 'o-', ms=2, alpha=0.5, color=f'C{i}')
        ax[0, 2].plot(t, G_p[i].numpy(), 's--', ms=2, alpha=0.7, color=f'C{i}')
    ax[0, 2].set(title='(c) $G/G_{max}$  (o=true, s=pred)', xlabel='Step')
    ax[0, 2].set_ylim(-0.05, 1.1)
    ax[0, 2].grid(True, alpha=0.3)

    # ── (d) Stiffness over time ──
    for i in range(ns):
        ax[0, 3].plot(t, K_t[i].numpy(), 'o-', ms=2, alpha=0.5, color=f'C{i}')
        ax[0, 3].plot(t, K_p[i].numpy(), 's--', ms=2, alpha=0.7, color=f'C{i}')
    ax[0, 3].set(title='(d) Stiffness $K$  (o=true, s=pred)', xlabel='Step')
    ax[0, 3].grid(True, alpha=0.3)

    # ── (e) Learned yield activation shapes ──
    sy, pp = model.slot_attn.yield_act.get_params()
    x_range = torch.linspace(-2, 2, 300)
    for s in range(N_SLOTS):
        sy_s = sy[s].mean()
        p_s = pp[s]
        ratio = (x_range / sy_s).abs().clamp(max=15.0)
        y = x_range / (1.0 + ratio.pow(p_s)).pow(1.0 / p_s)
        ax[1, 0].plot(x_range.numpy(), y.numpy(), lw=2,
                      label=f'Slot {s}: $\\sigma_y$={sy_s:.2f}, p={p_s:.1f}')
    ax[1, 0].axhline(0, color='k', lw=0.5)
    ax[1, 0].axvline(0, color='k', lw=0.5)
    ax[1, 0].set(title='(e) Learned Yield Activations', xlabel='x', ylabel='f(x)')
    ax[1, 0].legend(fontsize=6)
    ax[1, 0].grid(True, alpha=0.3)

    # ── (f) Slot attention heatmap ──
    attn = model.last_attn   # (B, K, T)
    if attn is not None:
        im = ax[1, 1].imshow(attn[0].numpy(), aspect='auto', cmap='YlOrRd',
                             interpolation='nearest')
        ax[1, 1].set_yticks(range(N_SLOTS))
        ax[1, 1].set_yticklabels([f'Slot {s}' for s in range(N_SLOTS)])
        plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    ax[1, 1].set(title='(f) Slot Attention Map', xlabel='Time step', ylabel='Slot')

    # ── (g) Predicted vs True displacement (scatter) ──
    ax[1, 2].scatter(u_t.flatten().numpy(), u_p.flatten().numpy(),
                     s=1, alpha=0.3, c='#1976D2')
    lim = max(u_t.max().item(), u_p.max().item()) * 1.05
    ax[1, 2].plot([0, lim], [0, lim], 'k--', lw=1, label='perfect')
    ax[1, 2].set(title='(g) $u$: Predicted vs True', xlabel='True', ylabel='Predicted')
    ax[1, 2].set_aspect('equal')
    ax[1, 2].legend(fontsize=7)
    ax[1, 2].grid(True, alpha=0.3)

    # ── (h) Smoothness check: histogram of Δu ──
    du_true = (u_t[:, 1:] - u_t[:, :-1]).flatten().numpy()
    du_pred = (u_p[:, 1:] - u_p[:, :-1]).flatten().numpy()
    ax[1, 3].hist(du_true, bins=30, alpha=0.5, label='True $\\Delta u$', color='#43A047')
    ax[1, 3].hist(du_pred, bins=30, alpha=0.5, label='Pred $\\Delta u$', color='#FF9800')
    ax[1, 3].set(title='(h) Smoothness: $\\Delta u$ Distribution', xlabel='$\\Delta u$')
    ax[1, 3].legend(fontsize=7)
    ax[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'slot_yield_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
if __name__ == '__main__':
    train()
