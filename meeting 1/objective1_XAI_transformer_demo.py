"""
Objective 1 – Physics-Informed XAI OWTTransformer (v4 – simplified)

Input  per step t (12):  L, D, e, H_{t-1}(3), M_{t-1}(2), G/Gmax_{t-1}, f0_{t-1}, u_t, θ_t
Output per step t  (7):  H_t(3), M_t(2), G/Gmax_t, f0_t

Flow:  Embedding → Transformer Encoder-Decoder → Output Heads
"""

# --- Core libraries ---
import torch, torch.nn as nn, torch.nn.functional as F  # PyTorch deep learning
import numpy as np, matplotlib
matplotlib.use("Agg")                                     # non-interactive backend (saves to file)
import matplotlib.pyplot as plt

# Reproducibility: fix random seeds so results are identical every run
torch.manual_seed(42); np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_SAMPLES, SEQ_LEN = 300, 20        # 300 synthetic scenarios, 20 typhoon load steps each
D_MODEL, N_HEADS   = 64, 4          # transformer hidden size & number of attention heads
N_LAYERS, FF_DIM   = 2, 128         # 2 encoder + 2 decoder layers; feed-forward width
EPOCHS, BATCH, LR   = 80, 32, 1e-3  # training: 80 epochs, batch size 32, learning rate 0.001
DOF, M_DIM          = 3, 2          # H has 3 DOF (K_L, K_R, K_LR); M has 2 DOF (M_lat, M_rot)

# Physics-informed loss weights (higher = more important)
# H, M, G, f0        → data-fitting MSE terms
# freq               → LoT1: f0 must be consistent with sqrt(K_R / M_rot)
# monoH, monoG       → LoT2: stiffness & G/Gmax must not increase over time
# pos                → LoT2: stiffness & mass must stay non-negative
# Grange             → G/Gmax must stay inside [0, 1]
W = dict(H=1., M=1., G=1., f0=1., freq=.5,
         monoH=.3, monoG=.3, pos=.3, Grange=.3)

# ── 1. Synthetic Data ──────────────────────────────────────────────────────
def make_data():
    """Generate synthetic time-series mimicking FEM + lab data for OWT.

    Physics modelled:
      - G/Gmax starts at 1.0 and degrades toward 0 (Vucetic & Dobry 1991)
      - Stiffness H drops proportionally to G/Gmax decay
      - f0 = 0.30 * sqrt(K_R / M_rot)  (Bhattacharya 2019)
    """
    N, T = N_SAMPLES, SEQ_LEN

    # --- Static pile geometry (constant across all time steps) ---
    L = torch.rand(N,1)*.8+.1        # pile length          ~ [0.1, 0.9]
    D = torch.rand(N,1)*.8+.1        # pile diameter         ~ [0.1, 0.9]
    e = torch.rand(N,1)*.6+.1        # eccentricity          ~ [0.1, 0.7]
    static = torch.cat([L, D, e], 1) # (N, 3)

    # --- Initial values at t=0 (before any typhoon loading) ---
    H0 = (.3*D + .25*L + .15).expand(-1, DOF) + .03*torch.randn(N, DOF)  # stiffness
    H0 = H0.clamp(min=.1)
    M0 = (.2*D + .1*L + .3).expand(-1, M_DIM) + .02*torch.randn(N, M_DIM)  # mass
    M0 = M0.clamp(min=.1)
    G0 = torch.ones(N)               # G/Gmax = 1.0 (undamaged soil)
    # f0 from physics: f0 ~ 0.30 * sqrt(K_R / M_rot), clamped to [0.1, 0.4] Hz
    f0_init = (.30*torch.sqrt((H0[:,1]/M0[:,1]).clamp(min=.01))).clamp(.1,.4)

    # --- Typhoon loading: lateral displacement u(t) and rotation θ(t) ---
    t_norm = torch.linspace(0,1,T).unsqueeze(0).expand(N,-1)  # normalized time
    u  = .5*t_norm + .05*torch.randn(N,T)   # lateral displacement (increases with time)
    th = .3*t_norm + .03*torch.randn(N,T)   # rotation (increases with time)

    # --- Allocate output tensors ---
    Hp, Mp, Gp = H0.clone(), M0.clone(), G0.clone()  # running state
    H_all  = torch.zeros(N, T, DOF)    # stiffness ground truth at each step
    M_all  = torch.zeros(N, T, M_DIM)  # mass ground truth
    G_all  = torch.zeros(N, T)         # G/Gmax ground truth
    f0_all = torch.zeros(N, T)         # natural frequency ground truth

    # --- Step through each typhoon load step ---
    for t in range(T):
        # Combined load magnitude: sqrt(u² + θ²)
        load = (u[:,t]**2 + th[:,t]**2).sqrt()

        # G/Gmax degrades with cyclic strain (Vucetic & Dobry 1991 style)
        Gp = (Gp - .012*load - .003*torch.randn(N).abs()).clamp(.05, 1.)

        # Compute ratio of current G/Gmax to previous step's value
        # This ratio drives proportional stiffness drop
        if t == 0:
            ratio_drop = (Gp / G0.clamp(min=.01)).unsqueeze(-1)
        else:
            ratio_drop = (Gp / G_all[:,t-1].clamp(min=.01)).unsqueeze(-1)

        # Stiffness drops proportionally to G/Gmax decay + small noise
        Hp = (Hp - (1 - ratio_drop)*Hp - .003*torch.randn_like(Hp).abs()).clamp(min=.01)

        # Mass evolves slightly (small decrease from loading + noise)
        Mp = (Mp - .001*load.unsqueeze(-1)*Mp + .002*torch.randn_like(Mp)).clamp(min=.05)

        # Natural frequency from physics: f0 = 0.30 * sqrt(K_R / M_rot)
        f0_t = (.30*torch.sqrt((Hp[:,1]/Mp[:,1]).clamp(min=.01))).clamp(.1,.4)

        # Store ground truth
        H_all[:,t], M_all[:,t], G_all[:,t], f0_all[:,t] = Hp, Mp, Gp, f0_t

    return dict(static=static, H=H_all, M=M_all, G=G_all, f0=f0_all,
                u=u, th=th, H0=H0, M0=M0, G0=G0, f0_init=f0_init)

# ── 2. Model ───────────────────────────────────────────────────────────────
def sinusoidal_pe(length, d):
    """Create sinusoidal positional encoding (Vaswani et al. 2017).

    Gives the transformer a sense of *order* — which step is t=0, t=1, etc.
    Even indices use sin, odd indices use cos, with exponentially decreasing
    frequencies so the model can distinguish nearby vs. far-apart steps.
    """
    pe = torch.zeros(1, length, d)                     # (1, T, d_model)
    pos = torch.arange(length).float().unsqueeze(1)    # (T, 1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.) / d))
    pe[0, :, 0::2] = torch.sin(pos * div)             # even dims
    pe[0, :, 1::2] = torch.cos(pos * div)             # odd dims
    return pe


class OWTTransformer(nn.Module):
    """Offshore Wind Turbine Transformer — encoder-decoder architecture.

    Diagram flow:
      Embedding (9 groups) → Feed-Forward projection → + Positional Encoding
      → Transformer Encoder (self-attention over load history)
      → Transformer Decoder (causal self-attn + cross-attn to encoder memory)
      → 4 parallel output heads → H_t, M_t, G/Gmax_t, f0_t
    """
    def __init__(self):
        super().__init__()
        emb = 16  # each input feature is embedded to 16-dim

        # --- Embedding: one small linear layer per input group ---
        # 9 groups × 16 = 144 concatenated, then projected → d_model (64)
        self.embs = nn.ModuleList([
            nn.Linear(1, emb),      # 0: L           (pile length)
            nn.Linear(1, emb),      # 1: D           (pile diameter)
            nn.Linear(1, emb),      # 2: e           (eccentricity)
            nn.Linear(DOF, emb),    # 3: H_{t-1}     (prev stiffness, 3 DOF)
            nn.Linear(M_DIM, emb),  # 4: M_{t-1}     (prev mass, 2 DOF)
            nn.Linear(1, emb),      # 5: G/Gmax_{t-1}(prev soil degradation)
            nn.Linear(1, emb),      # 6: f0_{t-1}    (prev natural frequency)
            nn.Linear(1, emb),      # 7: u_t         (lateral displacement)
            nn.Linear(1, emb),      # 8: theta_t     (rotation)
        ])
        # Feed-forward: concat all 9 embeddings → project to d_model
        self.proj = nn.Sequential(nn.Linear(9*emb, D_MODEL), nn.ReLU(),
                                  nn.Linear(D_MODEL, D_MODEL))
        # Fixed sinusoidal positional encoding (not learned)
        self.pe = nn.Parameter(sinusoidal_pe(SEQ_LEN, D_MODEL), requires_grad=False)

        # --- Encoder: standard transformer encoder (self-attention) ---
        enc_layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, FF_DIM,
                                               dropout=.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, N_LAYERS)

        # --- Decoder: custom layer (self-attn → cross-attn → FFN) ---
        # Decoder input = previous outputs [H, M, G/Gmax, f0] = 3+2+1+1 = 7 dims
        self.dec_proj = nn.Linear(DOF + M_DIM + 2, D_MODEL)  # 7 → d_model
        self.dec_self  = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=.1, batch_first=True)
        self.dec_cross = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=.1, batch_first=True)
        self.dec_ffn   = nn.Sequential(nn.Linear(D_MODEL, FF_DIM), nn.ReLU(),
                                       nn.Linear(FF_DIM, D_MODEL))
        self.dec_norms = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(3)])  # 3 LayerNorms

        # --- 4 parallel output heads ---
        # Softplus → ensures H, M are always positive (physical constraint)
        self.head_H  = nn.Sequential(nn.Linear(D_MODEL,64), nn.ReLU(),
                                     nn.Linear(64, DOF), nn.Softplus())    # → H_t (3)
        self.head_M  = nn.Sequential(nn.Linear(D_MODEL,64), nn.ReLU(),
                                     nn.Linear(64, M_DIM), nn.Softplus())  # → M_t (2)
        # Sigmoid → G/Gmax naturally bounded to [0, 1]
        self.head_G  = nn.Sequential(nn.Linear(D_MODEL,32), nn.ReLU(),
                                     nn.Linear(32,1), nn.Sigmoid())        # → G/Gmax_t (1)
        # f0: raw output passed through scaled sigmoid in forward()
        self.head_f0 = nn.Sequential(nn.Linear(D_MODEL,32), nn.ReLU(),
                                     nn.Linear(32,1))                      # → f0_t (1)

        self.cross_attn_w = None   # stored for XAI (explainability)

    def forward(self, static, H_prev, M_prev, G_prev, f0_prev, u, th):
        """
        Args:
            static  : (B, 3)    pile geometry [L, D, e]
            H_prev  : (B, T, 3) previous stiffness (autoregressive)
            M_prev  : (B, T, 2) previous mass
            G_prev  : (B, T)    previous G/Gmax
            f0_prev : (B, T)    previous natural frequency
            u       : (B, T)    lateral displacement at each step
            th      : (B, T)    rotation at each step
        Returns:
            H (B,T,3), M (B,T,2), G (B,T), f0 (B,T)
        """
        B, T = u.shape

        # --- 1. Embedding: broadcast static features to every time step ---
        parts = [static[:, i:i+1].unsqueeze(1).expand(B,T,1) for i in range(3)]  # L, D, e
        parts += [H_prev, M_prev,                           # autoregressive inputs
                  G_prev.unsqueeze(-1), f0_prev.unsqueeze(-1),
                  u.unsqueeze(-1), th.unsqueeze(-1)]         # external loading
        # Apply each embedding layer to its corresponding input, then concatenate
        cat = torch.cat([e(p) for e, p in zip(self.embs, parts)], -1)  # (B, T, 9×16=144)

        # --- 2. Project to d_model and add positional encoding ---
        enc_in = self.proj(cat) + self.pe[:, :T]            # (B, T, 64)

        # --- 3. Encoder: self-attention over the full loading history ---
        memory = self.encoder(enc_in)                        # (B, T, 64)

        # --- 4. Decoder ---
        # Decoder input = concatenation of all previous-step outputs (autoregressive)
        dec_tgt = torch.cat([H_prev, M_prev, G_prev.unsqueeze(-1),
                             f0_prev.unsqueeze(-1)], -1)     # (B, T, 7)
        x = self.dec_proj(dec_tgt) + self.pe[:, :T]          # project 7 → 64

        # Causal mask: decoder can only attend to past steps (not future)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(u.device)

        # Self-attention (decoder queries itself)
        a, _ = self.dec_self(x, x, x, attn_mask=mask)
        x = self.dec_norms[0](x + a)                         # residual + LayerNorm

        # Cross-attention (decoder queries encoder memory — the key XAI signal)
        a, cw = self.dec_cross(x, memory, memory, need_weights=True)
        self.cross_attn_w = cw.detach()                      # save for XAI visualization
        x = self.dec_norms[1](x + a)                         # residual + LayerNorm

        # Feed-forward sub-layer
        x = self.dec_norms[2](x + self.dec_ffn(x))           # residual + LayerNorm

        # --- 5. Output heads ---
        H  = self.head_H(x)                                  # (B, T, 3) stiffness
        M  = self.head_M(x)                                  # (B, T, 2) mass
        G  = self.head_G(x).squeeze(-1)                      # (B, T)   G/Gmax ∈ [0,1]
        # Scaled sigmoid: maps raw output to [0.1, 0.4] Hz (typical OWT range)
        f0 = .1 + .3*torch.sigmoid(self.head_f0(x).squeeze(-1))  # (B, T) f0
        return H, M, G, f0

# ── 3. Physics Loss ────────────────────────────────────────────────────────
def physics_loss(Hp, Mp, Gp, fp, Ht, Mt, Gt, ft):
    """9-term physics-informed loss function.

    Terms 1-4: standard MSE (data fitting)
    Term 5   : LoT1 — f0 must agree with sqrt(K_R / M_rot)  (physics consistency)
    Terms 6-7: LoT2 — stiffness & G/Gmax must monotonically decrease (degradation)
    Term 8   : LoT2 — stiffness & mass must be non-negative
    Term 9   : G/Gmax must stay in [0, 1]  (it's a ratio)
    """
    # --- Data-fitting MSE losses ---
    L_H  = F.mse_loss(Hp, Ht)    # stiffness error
    L_M  = F.mse_loss(Mp, Mt)    # mass error
    L_G  = F.mse_loss(Gp, Gt)    # G/Gmax error
    L_f0 = F.mse_loss(fp, ft)    # natural frequency error

    # --- LoT1: f0 ↔ stiffness consistency ---
    # Compute what f0 *should* be from predicted K_R and M_rot
    f0_phys = (.30*torch.sqrt((Hp[:,:,1]/Mp[:,:,1]).clamp(min=.01))).clamp(.1,.4)
    L_freq = F.mse_loss(fp, f0_phys.detach())   # detach to avoid double gradient

    # --- LoT2: monotonic degradation (penalize any *increase*) ---
    L_monoH  = F.relu(Hp[:,1:] - Hp[:,:-1]).mean()   # stiffness must not go up
    L_monoG  = F.relu(Gp[:,1:] - Gp[:,:-1]).mean()   # G/Gmax must not go up

    # --- LoT2: non-negativity (penalize any negative values) ---
    L_pos    = F.relu(-Hp).mean() + F.relu(-Mp).mean()

    # --- G/Gmax range constraint: must stay in [0, 1] ---
    L_Grange = F.relu(Gp - 1).mean() + F.relu(-Gp).mean()

    # --- Weighted sum ---
    total = (W['H']*L_H + W['M']*L_M + W['G']*L_G + W['f0']*L_f0
             + W['freq']*L_freq + W['monoH']*L_monoH + W['monoG']*L_monoG
             + W['pos']*L_pos + W['Grange']*L_Grange)
    return total, dict(total=total, H=L_H, M=L_M, G=L_G, f0=L_f0,
                       freq=L_freq, monoH=L_monoH, monoG=L_monoG, pos=L_pos)

# ── 4. Helpers ──────────────────────────────────────────────────────────────
def make_batch(data, idx):
    """Build autoregressive input/target pairs for a batch of sample indices.

    For each variable (H, M, G, f0), the *input* is shifted by one step:
      input  at step t = ground-truth at step t-1  (teacher forcing)
      target at step t = ground-truth at step t
    At t=0 the input is the initial value (H0, M0, G0=1, f0_init).

    Returns a flat tuple:
      (static, H_prev, M_prev, G_prev, f0_prev, u, th,  # 7 model inputs
       H_true, M_true, G_true, f0_true)                  # 4 targets
    """
    H, M, G, f0 = data['H'][idx], data['M'][idx], data['G'][idx], data['f0'][idx]
    # Shift right: prepend initial value, drop last step
    H_prev = torch.cat([data['H0'][idx].unsqueeze(1), H[:,:-1]], 1)
    M_prev = torch.cat([data['M0'][idx].unsqueeze(1), M[:,:-1]], 1)
    G_prev = torch.cat([data['G0'][idx].unsqueeze(1), G[:,:-1]], 1)
    f_prev = torch.cat([data['f0_init'][idx].unsqueeze(1), f0[:,:-1]], 1)
    return (data['static'][idx], H_prev, M_prev, G_prev, f_prev,
            data['u'][idx], data['th'][idx], H, M, G, f0)

def r2(p, t):
    """R² (coefficient of determination). 1.0 = perfect, <0 = worse than mean."""
    return (1 - ((t-p)**2).sum() / ((t-t.mean())**2).sum()).item()

def mape(p, t):
    """Mean Absolute Percentage Error (%)."""
    return ((t-p).abs() / t.clamp(min=.01)).mean().item() * 100

# ── 5. Train ────────────────────────────────────────────────────────────────
def train():
    """Full pipeline: generate data → build model → train → test → visualize."""
    sep = "=" * 65
    print(f"{sep}\nObjective 1 – OWTTransformer v4 (simplified)\n{sep}")

    # Step 1: Generate synthetic dataset
    data = make_data()
    n_tr = int(.85 * N_SAMPLES)                          # 85% train, 15% test
    tr, te = torch.arange(n_tr), torch.arange(n_tr, N_SAMPLES)

    # Step 2: Build model, optimizer, scheduler
    model = OWTTransformer()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)  # Adam optimizer
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)  # LR decay
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training {EPOCHS} epochs  ({n_tr} train / {len(te)} test)\n")

    # Track all 9 loss components per epoch
    keys = ['total','H','M','G','f0','freq','monoH','monoG','pos']
    hist = {k: [] for k in keys}

    # Step 3: Training loop
    for ep in range(EPOCHS):
        model.train()                                     # enable dropout
        perm = torch.randperm(n_tr)                       # shuffle training data
        sums = {k: 0. for k in keys}; nb = 0

        # Mini-batch loop
        for i in range(0, n_tr, BATCH):
            b = make_batch(data, perm[i:i+BATCH])         # build autoregressive batch
            Hp, Mp, Gp, fp = model(*b[:7])                # forward pass (7 inputs)
            loss, losses = physics_loss(Hp, Mp, Gp, fp, *b[7:])  # 9-term loss
            opt.zero_grad(); loss.backward()              # backpropagation
            nn.utils.clip_grad_norm_(model.parameters(), 1.)  # gradient clipping
            opt.step()                                    # update weights
            for k in keys: sums[k] += losses[k].item()
            nb += 1

        # Record epoch-average losses
        for k in keys: hist[k].append(sums[k]/nb)
        sched.step()                                      # decay learning rate

        # Print progress every 10 epochs
        if (ep+1) % 10 == 0 or ep == 0:
            h = hist
            print(f"  Ep {ep+1:3d}/{EPOCHS}  Tot:{h['total'][-1]:.4f}  "
                  f"H:{h['H'][-1]:.4f}  M:{h['M'][-1]:.4f}  "
                  f"G:{h['G'][-1]:.4f}  f0:{h['f0'][-1]:.4f}")

    # Step 4: Evaluate on held-out test set
    print("\n--- Test ---")
    model.eval()                                          # disable dropout
    with torch.no_grad():                                 # no gradient computation
        b = make_batch(data, te)
        Hp, Mp, Gp, fp = model(*b[:7])
        Ht, Mt, Gt, ft = b[7:]
        # Print MSE, R², MAPE for each output
        for name, p, t in [("H", Hp, Ht), ("M", Mp, Mt),
                           ("G/Gmax", Gp, Gt), ("f0", fp, ft)]:
            print(f"  {name:7s}  MSE={F.mse_loss(p,t):.6f}  "
                  f"R²={r2(p,t):.4f}  MAPE={mape(p,t):.2f}%")

    # Step 5: Generate 10-panel visualization
    visualize(model, data, te, hist)
    print(f"\n{sep}\nDone. Saved → objective1_XAI_results.png\n{sep}")
    return model

# ── 6. Visualization ───────────────────────────────────────────────────────
def visualize(model, data, te, hist):
    """Generate a 2×5 = 10-panel figure summarizing all results.

    Panels:
      (a) Training loss curves (log scale)     (f) f0: true vs predicted
      (b) K_L: true vs predicted               (g) Predicted H(t) degradation
      (c) K_R: true vs predicted               (h) G/Gmax degradation curve
      (d) M_rot: true vs predicted             (i) Decoder cross-attention heatmap
      (e) G/Gmax: true vs predicted            (j) Input attribution bar chart
    """
    fig, ax = plt.subplots(2, 5, figsize=(26, 10))
    fig.suptitle("OWTTransformer v4 – Physics-Informed XAI Demo", fontsize=13)
    t = np.arange(SEQ_LEN)  # x-axis: typhoon load steps 0..19

    # Get test predictions
    model.eval()
    with torch.no_grad():
        b = make_batch(data, te)
        Hp, Mp, Gp, fp = model(*b[:7])
    Ht, Mt, Gt, ft = b[7:]
    ns = min(4, len(te))  # plot up to 4 test scenarios

    # (a) losses
    for k, ls in [('total','-'),('H','--'),('M','--'),('G','-.'),
                  ('f0','-.'),('freq',':'),('monoH',':'),('monoG',':')]:
        ax[0,0].plot(hist[k], label=k, ls=ls)
    ax[0,0].set(title='(a) Losses', yscale='log'); ax[0,0].legend(fontsize=5); ax[0,0].grid(True, alpha=.3)

    # (b-d) K_L, K_R, M_rot
    for col, j, lbl in [(1,0,'$K_L$'),(2,1,'$K_R$')]:
        for i in range(ns):
            ax[0,col].plot(t, Ht[i,:,j], 'o-', ms=2, alpha=.5, color=f'C{i}')
            ax[0,col].plot(t, Hp[i,:,j], 's--', ms=2, alpha=.7, color=f'C{i}')
        ax[0,col].set_title(f'({chr(98+col-1)}) {lbl}'); ax[0,col].grid(True, alpha=.3)

    for i in range(ns):
        ax[0,3].plot(t, Mt[i,:,1], 'o-', ms=2, alpha=.5, color=f'C{i}')
        ax[0,3].plot(t, Mp[i,:,1], 's--', ms=2, alpha=.7, color=f'C{i}')
    ax[0,3].set_title('(d) $M_{rot}$'); ax[0,3].grid(True, alpha=.3)

    # (e) G/Gmax
    for i in range(ns):
        ax[0,4].plot(t, Gt[i], 'o-', ms=2, alpha=.5, color=f'C{i}')
        ax[0,4].plot(t, Gp[i], 's--', ms=2, alpha=.7, color=f'C{i}')
    ax[0,4].set_title('(e) $G/G_{max}$'); ax[0,4].set_ylim(-.05,1.1); ax[0,4].grid(True, alpha=.3)

    # (f) f0
    for i in range(ns):
        ax[1,0].plot(t, ft[i], 'o-', ms=2, alpha=.5, color=f'C{i}')
        ax[1,0].plot(t, fp[i], 's--', ms=2, alpha=.7, color=f'C{i}')
    ax[1,0].set_title('(f) $f_0$ (Hz)'); ax[1,0].grid(True, alpha=.3)

    # (g) H degradation single scenario
    for j, lbl in enumerate(['$K_L$','$K_R$','$K_{LR}$']):
        ax[1,1].plot(t, Hp[0,:,j], label=lbl, lw=2)
    ax[1,1].set_title('(g) Pred. H(t)'); ax[1,1].legend(fontsize=7); ax[1,1].grid(True, alpha=.3)

    # (h) G/Gmax single scenario
    ax[1,2].plot(t, Gt[0], 'o-', label='True', ms=3)
    ax[1,2].plot(t, Gp[0], 's--', label='Pred', ms=3)
    ax[1,2].set_title('(h) G/Gmax Curve'); ax[1,2].legend(); ax[1,2].set_ylim(-.05,1.1); ax[1,2].grid(True, alpha=.3)

    # (i) Cross-attention heatmap — XAI: shows which encoder tokens
    #     the decoder attends to when making predictions
    cw = model.cross_attn_w
    if cw is not None:
        im = ax[1,3].imshow(cw[0].numpy(), aspect='auto', cmap='RdYlBu_r')
        plt.colorbar(im, ax=ax[1,3], fraction=.046)
    ax[1,3].set_title('(i) Cross-Attention')

    # (j) Attribution bar — XAI: average attention score per encoder token
    #     Higher bar = that time step's input features influenced predictions more
    if cw is not None:
        attr = cw.mean(dim=(0,1)).numpy()  # average over batch & decoder queries
        ax[1,4].bar(range(len(attr)), attr, color='#FF9800', alpha=.85)
    ax[1,4].set_title('(j) Attribution'); ax[1,4].grid(True, alpha=.3, axis='y')

    plt.tight_layout()
    plt.savefig('objective1_XAI_results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    train()
