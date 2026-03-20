"""
=============================================================================
Thermodynamic 21-Slot Model: Load-Driven Soil Degradation
=============================================================================
Based on hand-sketch physics:
  - S1 = G0, H, e  (initial: soil shear modulus, horizontal load, porosity)
  - S2 - S1 = 4*H*i[j]_{n=1}  (state change driven by load history)
  
Architecture:
  21 specialized slots, organized by domain & load sensitivity:
  
  GROUP 1 (Slot 0-6): LOAD ANALYSIS (7 slots)
    - Horizontal load (H) evolution & detection
    - Moment (M) evolution & detection  
    - Cyclic/cumulative load tracking
    - Load-triggered yield detection
  
  GROUP 2 (Slot 7-13): SOIL DEGRADATION (7 slots)
    - Shear modulus (G0 → G/Gmax) degradation
    - Stiffness ratio (Cd/Cdo) tracking
    - Cyclic strain accumulation
    - Energy dissipation (thermodynamic term 1)
    - Entropy/irreversibility (thermodynamic term 2)
  
  GROUP 3 (Slot 14-19): PILE & STRUCTURAL (6 slots)
    - Pile properties (L, D, EI) response
    - Soil-pile interaction
    - Displacement accumulation
    - Stiffness drop when load increases
  
  GROUP 4 (Slot 20): FUSION (1 slot)
    - Combine all 21 slots
    - Cross-attention over domains
    - Output: primary target is G/Gmax prediction

Physics principle:
  Stiffness C decreases monotonically as shear strain γ increases:
    C(γ) = C_max / (1 + (γ/γ_ref)^α)
  where γ increases with load (≈ proportional to H, M, N_cycles).
  
  Thermodynamic constraint:
    dE_dissipated ≥ 0  (1st law: energy balance)
    dS ≥ dQ/T         (2nd law: entropy increase)

Sources:
  - Hardin & Drnevich (1972): G/Gmax vs cyclic strain
  - Vucetic & Dobry (1991): degradation curves
  - DNV-OS-J101: pile-soil interaction
  - Thermodynamics of irreversible processes
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
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")  # local data cache for reproducible runs
os.makedirs(DATA_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
D_MODEL     = 48          # hidden dimension
N_SLOTS     = 21          # 7 load + 7 soil + 6 pile + 1 fusion
SLOT_ITERS  = 3           # fusion refinement iterations
EPOCHS      = 250
BATCH       = 32
LR          = 2e-3

# Feature dimensions per CSV
FEAT_PILE   = 4   # L_m, D_m, EI_Nm2, wall_thickness
FEAT_SOIL   = 5   # G0, Su, void_ratio, OCR, degradation_factor
FEAT_LOAD   = 6   # H_horiz, M_moment, N_cycles, freq, cum_strain_H, cum_strain_M
FEAT_ENV    = 3   # wave_H, wind_V, water_depth
FEAT_SLOT0  = 2 + FEAT_PILE + FEAT_SOIL  # [H, M] + pile + soil -> 11 features


# ============================================================================
# 1. THERMODYNAMIC DATA GENERATOR — Load-driven degradation
# ============================================================================
def generate_thermodynamic_data():
    """
    Generate training data that embeds thermodynamic principles:
      - Stiffness C decreases with cumulative load
      - G/Gmax follows Hardin-Drnevich degradation
      - Energy dissipation increases monotonically
      - Entropy increases (irreversibility)
    """
    N_SCENARIOS = 200
    N_STEPS = 30
    
    data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\nGenerating thermodynamic training data ({N_SCENARIOS} scenarios × {N_STEPS} steps)...")
    
    # Scenario enumeration
    scenario_ids = np.repeat(np.arange(N_SCENARIOS), N_STEPS)
    steps = np.tile(np.arange(N_STEPS), N_SCENARIOS)
    
    # ── Pile properties (constant per scenario) ──
    L_pile = np.random.uniform(30, 60, N_SCENARIOS)      # pile length [m]
    D_pile = np.random.uniform(3, 6, N_SCENARIOS)        # diameter [m]
    EI = np.random.uniform(1e9, 5e9, N_SCENARIOS)        # bending stiffness
    t_wall = np.random.uniform(0.05, 0.15, N_SCENARIOS) * D_pile
    
    pile_data = []
    for s in scenario_ids:
        pile_data.append([L_pile[s], D_pile[s], EI[s], t_wall[s]])
    pile_df = pd.DataFrame(pile_data, columns=['L_m', 'D_m', 'EI_Nm2', 'wall_thickness'])
    
    # ── Soil properties (constant per scenario, but sampled) ──
    G0 = np.random.uniform(50, 150, N_SCENARIOS)         # initial shear modulus [MPa]
    Su = np.random.uniform(30, 100, N_SCENARIOS)         # undrained strength [kPa]
    void_ratio = np.random.uniform(0.6, 0.9, N_SCENARIOS)
    OCR = np.random.uniform(1.5, 4.0, N_SCENARIOS)
    deg_factor = np.random.uniform(0.8, 1.2, N_SCENARIOS)
    
    soil_data = []
    for s in scenario_ids:
        soil_data.append([G0[s], Su[s], void_ratio[s], OCR[s], deg_factor[s]])
    soil_df = pd.DataFrame(soil_data, columns=['G0_MPa', 'Su_kPa', 'void_ratio', 'OCR', 'deg_factor'])
    
    # ── Load evolution (increases over time) ──
    H_base = np.random.uniform(100, 500, N_SCENARIOS)    # baseline horizontal force [kN]
    M_base = np.random.uniform(1000, 5000, N_SCENARIOS)  # baseline moment [kNm]
    freq_load = np.random.uniform(0.5, 2.0, N_SCENARIOS) # loading frequency
    
    load_data = []
    cum_strain_H = []
    cum_strain_M = []
    
    for s in range(N_SCENARIOS):
        for t in range(N_STEPS):
            # Load ramps up over time (storm intensifies)
            time_factor = (t + 1) / N_STEPS  # 0.033 to 1.0
            H_t = H_base[s] * time_factor * (1 + 0.2 * np.sin(2*np.pi*freq_load[s]*t/N_STEPS))
            M_t = M_base[s] * time_factor * (1 + 0.2 * np.sin(2*np.pi*freq_load[s]*t/N_STEPS))
            N_cycles_t = int(t * freq_load[s])
            
            # Cumulative strain (proportional to load path)
            cum_strain_H_t = sum([H_base[s] * (i+1)/N_STEPS for i in range(t+1)]) / (L_pile[s] * G0[s])
            cum_strain_M_t = sum([M_base[s] * (i+1)/N_STEPS for i in range(t+1)]) / (EI[s] * G0[s] + 1e-6)
            
            load_data.append([H_t, M_t, N_cycles_t, freq_load[s], cum_strain_H_t, cum_strain_M_t])
            cum_strain_H.append(cum_strain_H_t)
            cum_strain_M.append(cum_strain_M_t)
    
    load_df = pd.DataFrame(load_data, 
                           columns=['H_horiz_kN', 'M_moment_kNm', 'N_cycles', 'freq_Hz', 'cum_strain_H', 'cum_strain_M'])
    
    # ── Environment (waves, wind) ──
    env_data = []
    for s in scenario_ids:
        wave_H = np.random.uniform(1, 8, 1)[0]
        wind_V = np.random.uniform(10, 40, 1)[0]
        water_d = np.random.uniform(20, 50, 1)[0]
        env_data.append([wave_H, wind_V, water_d])
    env_df = pd.DataFrame(env_data, columns=['wave_H_m', 'wind_V_ms', 'water_depth_m'])
    
    # ── GROUND TRUTH: Degradation-based on load ──
    # G/Gmax follows Hardin-Drnevich: G/Gmax = 1 / (1 + (γ/γ_ref)^α)
    # where γ = cumulative strain from load
    
    gt_data = []
    for i in range(len(scenario_ids)):
        s = scenario_ids[i]
        gamma = cum_strain_H[i] + cum_strain_M[i]  # total cyclic strain (normalized)
        
        # Hardin-Drnevich degradation
        gamma_ref = 0.1 * deg_factor[s]
        alpha = 0.9
        G_ratio = 1.0 / (1.0 + (gamma / gamma_ref) ** alpha)
        G_ratio = np.clip(G_ratio, 0.1, 1.0)
        
        # Stiffness ratio: drops as load increases
        cum_load_norm = (cum_strain_H[i] + cum_strain_M[i]) / 10.0  # normalize
        C_ratio = G_ratio * np.exp(-0.5 * cum_load_norm)
        C_ratio = np.clip(C_ratio, 0.05, 1.0)
        
        # Energy dissipation (increases monotonically)
        E_diss = 1.0 - G_ratio  # dissipation ~ stiffness loss
        
        # Entropy change (2nd law)
        S_entropy = -np.log(np.clip(G_ratio, 0.01, 1.0))
        
        gt_data.append([G_ratio, C_ratio, E_diss, S_entropy])
    
    gt_df = pd.DataFrame(gt_data, columns=['G_ratio', 'C_ratio', 'E_dissipated', 'entropy_change'])
    
    # Save
    pile_df.to_csv(os.path.join(data_dir, 'pile.csv'), index=False)
    soil_df.to_csv(os.path.join(data_dir, 'soil.csv'), index=False)
    load_df.to_csv(os.path.join(data_dir, 'load.csv'), index=False)
    env_df.to_csv(os.path.join(data_dir, 'environment.csv'), index=False)
    gt_df.to_csv(os.path.join(data_dir, 'ground_truth.csv'), index=False)
    
    print(f"  ✓ pile.csv, soil.csv, load.csv, environment.csv, ground_truth.csv")
    
    # Return as tensors
    return {
        'pile': torch.tensor(pile_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'soil': torch.tensor(soil_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'load': torch.tensor(load_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'environment': torch.tensor(env_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'targets': torch.tensor(gt_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'n_scenarios': N_SCENARIOS,
        'n_steps': N_STEPS,
    }


def load_data():
    """Load cached CSVs or generate them if missing."""
    if not os.path.exists(os.path.join(DATA_DIR, 'ground_truth.csv')):
        return generate_thermodynamic_data()
    
    # Load existing
    N_SCENARIOS = 200
    N_STEPS = 30
    
    pile_df = pd.read_csv(os.path.join(DATA_DIR, 'pile.csv'))
    soil_df = pd.read_csv(os.path.join(DATA_DIR, 'soil.csv'))
    load_df = pd.read_csv(os.path.join(DATA_DIR, 'load.csv'))
    env_df = pd.read_csv(os.path.join(DATA_DIR, 'environment.csv'))
    gt_df = pd.read_csv(os.path.join(DATA_DIR, 'ground_truth.csv'))
    
    return {
        'pile': torch.tensor(pile_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'soil': torch.tensor(soil_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'load': torch.tensor(load_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'environment': torch.tensor(env_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'targets': torch.tensor(gt_df.values, dtype=torch.float32).reshape(N_SCENARIOS, N_STEPS, -1),
        'n_scenarios': N_SCENARIOS,
        'n_steps': N_STEPS,
    }


# ============================================================================
# 2. YIELD ACTIVATION (Physics-informed)
# ============================================================================
class YieldActivation(nn.Module):
    """Smooth yield surface: linear below threshold, saturates above.

    The learnable parameters (sigma_y, p) adapt the yield curve to each slot.
    """
    def __init__(self, dim):
        super().__init__()
        self.sigma_y_raw = nn.Parameter(torch.randn(dim) * 0.1)
        self.p_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        sigma_y = F.softplus(self.sigma_y_raw) + 0.01
        p = 1.5 + F.softplus(self.p_raw)
        ratio = (x / sigma_y).abs().clamp(max=15.0)
        return x / (1.0 + ratio.pow(p)).pow(1.0 / p)

    def get_params(self):
        sigma_y = F.softplus(self.sigma_y_raw) + 0.01
        p = 1.5 + F.softplus(self.p_raw)
        return sigma_y.detach(), p.detach()


# ============================================================================
# 3. SPECIALIZED SLOT (21 of these)
# ============================================================================
class SpecializedSlot(nn.Module):
    """Individual domain slot: embed -> self-attn -> yield activation."""
    def __init__(self, n_features, name="slot"):
        super().__init__()
        self.name = name
        
        self.embed = nn.Sequential(
            nn.Linear(n_features, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        
        self.self_attn = nn.MultiheadAttention(
            D_MODEL, num_heads=2, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(D_MODEL)
        
        self.yield_act = YieldActivation(D_MODEL)

    def forward(self, x):
        h = self.embed(x)
        h_n = self.norm(h)
        h_sa, _ = self.self_attn(h_n, h_n, h_n)
        h = h + h_sa
        h = self.yield_act(h)
        return h


# ============================================================================
# 4. FUSION SLOT (combines all 21 slots)
# ============================================================================
class FusionSlot(nn.Module):
    """Fusion across all domain slots using cross-attention + GRU refinement."""
    def __init__(self, n_slots=21):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            D_MODEL, num_heads=2, batch_first=True, dropout=0.1)
        self.norm_q = nn.LayerNorm(D_MODEL)
        self.norm_kv = nn.LayerNorm(D_MODEL)
        
        self.fusion_query = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.gru = nn.GRUCell(D_MODEL, D_MODEL)
        
        self.yield_act = YieldActivation(D_MODEL)
        
        self.proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )

    def forward(self, slot_outputs):
        """slot_outputs: list of N tensors (B, T, D)"""
        B, T, D = slot_outputs[0].shape
        
        stacked = torch.stack(slot_outputs, dim=2)  # (B, T, N_slots, D)
        kv_flat = stacked.reshape(B * T, len(slot_outputs), D)
        q_flat = self.fusion_query.expand(B * T, -1, -1).clone()
        
        for _ in range(SLOT_ITERS):
            q_n = self.norm_q(q_flat)
            kv_n = self.norm_kv(kv_flat)
            out, aw = self.cross_attn(q_n, kv_n, kv_n)
            q_flat = self.gru(out.squeeze(1), q_flat.squeeze(1)).unsqueeze(1)
        
        fused = q_flat.squeeze(1).reshape(B, T, D)
        attn_w = aw.squeeze(1).reshape(B, T, len(slot_outputs))
        
        fused = self.yield_act(fused)
        fused = self.proj(fused)
        
        return fused, attn_w


# ============================================================================
# 5. FULL 21-SLOT MODEL
# ============================================================================
def sinusoidal_pe(length, d):
    """Deterministic sinusoidal positional encoding (no learned params)."""
    pe = torch.zeros(1, length, d)
    pos = torch.arange(length).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


class Thermodynamic21SlotModel(nn.Module):
    """21 specialized slots + 1 fusion slot."""
    def __init__(self, n_steps=30):
        super().__init__()

        # Slot 0: dedicated Gmax predictor from horizontal load, moment, pile, and soil
        self.slot0_gmax_hm = SpecializedSlot(FEAT_SLOT0, "Slot0-Gmax-HM-PileSoil")
        self.slot0_gmax_head = nn.Linear(D_MODEL, 1)
        self.slot0_trigger_head = nn.Linear(D_MODEL, 1)
        
        # ── 7 LOAD ANALYSIS SLOTS ──
        self.slot_H_low = SpecializedSlot(FEAT_LOAD, "H-Low")
        self.slot_H_mid = SpecializedSlot(FEAT_LOAD, "H-Mid")
        self.slot_H_high = SpecializedSlot(FEAT_LOAD, "H-High")
        self.slot_M_load = SpecializedSlot(FEAT_LOAD, "M-Load")
        self.slot_cyclic = SpecializedSlot(FEAT_LOAD, "Cyclic")
        self.slot_cum_H = SpecializedSlot(FEAT_LOAD, "CumH")
        self.slot_cum_M = SpecializedSlot(FEAT_LOAD, "CumM")
        
        # ── 7 SOIL DEGRADATION SLOTS ──
        self.slot_stiff = SpecializedSlot(FEAT_SOIL, "Stiffness")
        self.slot_strain = SpecializedSlot(FEAT_SOIL, "StrainAccum")
        self.slot_energy = SpecializedSlot(FEAT_SOIL, "EnergyDiss")
        self.slot_entropy = SpecializedSlot(FEAT_SOIL, "Entropy")
        self.slot_Su = SpecializedSlot(FEAT_SOIL, "Strength")
        self.slot_void = SpecializedSlot(FEAT_SOIL, "VoidRatio")
        
        # ── 6 PILE/STRUCTURAL SLOTS ──
        self.slot_pile_geom = SpecializedSlot(FEAT_PILE, "PileGeom")
        self.slot_pile_EI = SpecializedSlot(FEAT_PILE, "PileEI")
        self.slot_interaction = SpecializedSlot(FEAT_PILE, "Interaction")
        self.slot_disp = SpecializedSlot(FEAT_PILE, "Displacement")
        self.slot_stiff_drop = SpecializedSlot(FEAT_PILE, "StiffDrop")
        self.slot_env_pile = SpecializedSlot(FEAT_ENV, "EnvPile")
        
        # ── FUSION SLOT (21st) ──
        self.fusion_slot = FusionSlot(n_slots=21)
        
        # Positional encoding
        self.pe = nn.Parameter(sinusoidal_pe(n_steps, D_MODEL), requires_grad=False)
        
        # Transformer
        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=2, dim_feedforward=D_MODEL*2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Stable slot aggregation (20 specialized slots)
        self.slot_gate = nn.Linear(D_MODEL * 20, D_MODEL)
        
        # Output heads (focus on G/Gmax as primary)
        head_in = D_MODEL * 2
        self.head_G_Gmax = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                         nn.Linear(D_MODEL, 1), nn.Sigmoid())
        self.head_C_ratio = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                          nn.Linear(D_MODEL, 1), nn.Sigmoid())
        self.head_E_diss = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                         nn.Linear(D_MODEL, 1), nn.Sigmoid())
        self.head_entropy = nn.Sequential(nn.Linear(head_in, D_MODEL), nn.ReLU(),
                                          nn.Linear(D_MODEL, 1))
        
        self.last_fusion_attn = None
        self.last_trigger = None

        def forward(self, pile, soil, load, env):
                """Forward pass with Slot 0 anchor + multi-slot fusion.

                Inputs:
                    pile: (B, T, 4), soil: (B, T, 5), load: (B, T, 6), env: (B, T, 3)
                Returns:
                    G_Gmax, C_ratio, E_diss, entropy, gmax_slot0, trigger_state
                """
        B, T, _ = pile.shape

        # Slot 0: physics anchor using [H, M] + pile + soil
        hm = load[:, :, :2]
        slot0_in = torch.cat([hm, pile, soil], dim=-1)
        h_slot0 = self.slot0_gmax_hm(slot0_in)
        gmax_slot0 = torch.sigmoid(self.slot0_gmax_head(h_slot0).squeeze(-1))
        trigger = torch.sigmoid(self.slot0_trigger_head(h_slot0).squeeze(-1))
        trigger_state = torch.cummax(trigger, dim=1)[0]
        self.last_trigger = trigger_state.detach()
        
        # Process through slots
        load_slots = [
            self.slot_H_low(load),
            self.slot_H_mid(load),
            self.slot_H_high(load),
            self.slot_M_load(load),
            self.slot_cyclic(load),
            self.slot_cum_H(load),
            self.slot_cum_M(load),
        ]
        
        soil_slots = [
            self.slot_stiff(soil),
            self.slot_strain(soil),
            self.slot_energy(soil),
            self.slot_entropy(soil),
            self.slot_Su(soil),
            self.slot_void(soil),
        ]
        
        pile_slots = [
            self.slot_pile_geom(pile),
            self.slot_pile_EI(pile),
            self.slot_interaction(pile),
            self.slot_disp(pile),
            self.slot_stiff_drop(pile),
            self.slot_env_pile(env),
        ]
        
        # 20 specialized slots total: Slot0 + (7 load) + (6 soil) + (6 pile)
        all_slots = [h_slot0] + load_slots + soil_slots + pile_slots
        
        # Fusion
        fused, fusion_attn = self.fusion_slot(all_slots)
        self.last_fusion_attn = fusion_attn.detach()
        
        # Aggregate
        slot_cat = torch.cat(all_slots, dim=-1)
        slot_agg = self.slot_gate(slot_cat)
        
        # Encoder mixes temporal context after fusion
        x = fused + self.pe[:, :T]
        x = self.encoder(x)
        
        # Output
        x_out = torch.cat([x, slot_agg], dim=-1)
        gmax_main = self.head_G_Gmax(x_out).squeeze(-1)
        G_Gmax = torch.clamp(0.65 * gmax_main + 0.35 * gmax_slot0, 0.0, 1.0)
        C_ratio = self.head_C_ratio(x_out).squeeze(-1)
        E_diss = self.head_E_diss(x_out).squeeze(-1)
        entropy = self.head_entropy(x_out).squeeze(-1)

        # Triggered behavior: after trigger, stiffness drops more aggressively
        C_ratio = torch.clamp(C_ratio * (1.0 - 0.20 * trigger_state), 0.0, 1.0)
        E_diss = torch.clamp(E_diss + 0.10 * trigger_state, 0.0, 1.0)
        
        return G_Gmax, C_ratio, E_diss, entropy, gmax_slot0, trigger_state


def visualize_results(model, data, te_idx, loss_hist, comp_hist):
    """Save training/evaluation graphs to a PNG file."""
    model.eval()
    with torch.no_grad():
        G_p, C_p, E_p, S_p, _, trig = model(
            data['pile'][te_idx],
            data['soil'][te_idx],
            data['load'][te_idx],
            data['environment'][te_idx],
        )

    tgt = data['targets'][te_idx]
    G_t, C_t, E_t, S_t = tgt[:, :, 0], tgt[:, :, 1], tgt[:, :, 2], tgt[:, :, 3]
    n_steps = data['n_steps']
    t = np.arange(n_steps)
    ns = min(4, len(te_idx))

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Thermodynamic 21-Slot Model Results", fontsize=14, fontweight='bold')

    # (a) total training loss
    ax[0, 0].plot(loss_hist, color='#1f77b4', lw=2)
    ax[0, 0].set_title('(a) Total Training Loss')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_yscale('log')
    ax[0, 0].grid(True, alpha=0.3)

    # (b) component losses
    for key, color in [('G', '#2ca02c'), ('C', '#ff7f0e'), ('E', '#d62728'), ('S', '#9467bd')]:
        ax[0, 1].plot(comp_hist[key], label=key, color=color)
    ax[0, 1].set_title('(b) Loss Components')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('MSE')
    ax[0, 1].set_yscale('log')
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(True, alpha=0.3)

    # (c) G/Gmax over time (primary target)
    for i in range(ns):
        lbl_true = 'True' if i == 0 else None
        lbl_pred = 'Pred' if i == 0 else None
        ax[0, 2].plot(t, G_t[i].cpu().numpy(), 'o-', ms=3, alpha=0.6, color=f'C{i}', label=lbl_true)
        ax[0, 2].plot(t, G_p[i].cpu().numpy(), 's--', ms=3, alpha=0.8, color=f'C{i}', label=lbl_pred)
    ax[0, 2].set_title('(c) G/Gmax vs Time (o=true, s=pred)')
    ax[0, 2].set_xlabel('Step')
    ax[0, 2].set_ylim(0, 1.05)
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(True, alpha=0.3)

    # (d) stiffness ratio over time
    for i in range(ns):
        lbl_true = 'True' if i == 0 else None
        lbl_pred = 'Pred' if i == 0 else None
        ax[1, 0].plot(t, C_t[i].cpu().numpy(), 'o-', ms=3, alpha=0.6, color=f'C{i}', label=lbl_true)
        ax[1, 0].plot(t, C_p[i].cpu().numpy(), 's--', ms=3, alpha=0.8, color=f'C{i}', label=lbl_pred)
    ax[1, 0].set_title('(d) C_ratio vs Time')
    ax[1, 0].set_xlabel('Step')
    ax[1, 0].set_ylim(0, 1.05)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(True, alpha=0.3)

    # (e) scatter primary output
    g_true_flat = G_t.flatten().cpu().numpy()
    g_pred_flat = G_p.flatten().cpu().numpy()
    ax[1, 1].scatter(g_true_flat, g_pred_flat, s=5, alpha=0.3, color='#2ca02c')
    lim = float(max(g_true_flat.max(), g_pred_flat.max())) * 1.02
    ax[1, 1].plot([0, lim], [0, lim], 'k--', lw=1)
    ax[1, 1].set_title('(e) G/Gmax Predicted vs True')
    ax[1, 1].set_xlabel('True')
    ax[1, 1].set_ylabel('Predicted')
    ax[1, 1].set_aspect('equal')
    ax[1, 1].grid(True, alpha=0.3)

    # (f) attention/trigger behavior
    if model.last_fusion_attn is not None:
        attn = model.last_fusion_attn.mean(dim=0).cpu().numpy().T
        im = ax[1, 2].imshow(attn, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax[1, 2].set_title('(f) Fusion Attention (20 Specialized Slots)')
        ax[1, 2].set_xlabel('Time Step')
        ax[1, 2].set_ylabel('Slot Index')
        cbar = plt.colorbar(im, ax=ax[1, 2], fraction=0.046)
        cbar.set_label('Attention weight')
    else:
        ax[1, 2].set_title('(f) Fusion Attention not available')

    # Overlay average trigger curve on top of attention panel
    if trig is not None:
        trig_mean = trig.mean(dim=0).cpu().numpy()
        ax_t = ax[1, 2].twinx()
        ax_t.plot(np.arange(len(trig_mean)), trig_mean, color='cyan', lw=2, label='Trigger')
        ax_t.set_ylabel('Trigger', color='cyan')
        ax_t.tick_params(axis='y', colors='cyan')
        ax_t.set_ylim(0.0, 1.05)
        ax_t.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, 'thermodynamic_21slot_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def print_results_legend():
    """Print terminal legend to interpret metrics and plots."""
    print(f"\n{'-' * 70}")
    print("RESULTS LEGEND")
    print(f"{'-' * 70}")
    print("Metrics:")
    print("  G/Gmax    -> Main target (soil stiffness state). 1=stiff, lower=degraded")
    print("  C_ratio   -> Lateral stiffness ratio. Should drop when load increases")
    print("  E_diss    -> Dissipated energy proxy. Should generally increase")
    print("  Entropy   -> Irreversibility proxy. Should generally increase")
    print("  MSE       -> Mean Squared Error (lower is better)")
    print("  Slot0-Gmax-> Direct Slot 0 quality for the primary target")
    print("\nExpected physical trend:")
    print("  Load up => G/Gmax down, C_ratio down, E_diss up, Entropy up")
    print("\nFigure legend:")
    print("  (a) Total loss over epochs")
    print("  (b) Component losses: G, C, E, S")
    print("  (c) G/Gmax vs step: circles=True, squares=Pred")
    print("  (d) C_ratio vs step: circles=True, squares=Pred")
    print("  (e) Scatter Pred vs True: closer to diagonal is better")
    print("  (f) Heatmap=attention weights, cyan line=trigger level")


# ============================================================================
# 6. TRAINING
# ============================================================================
def train():
    """Train model, evaluate, then print legend and save plots."""
    print("\n" + "="*70)
    print("THERMODYNAMIC 21-SLOT MODEL: Load-Driven Soil Degradation")
    print("="*70)
    print("\nArchitecture:")
    print("  Slot 0:      Gmax from H+M+pile+soil (with trigger)")
    print("  Slots 1-7:   Load Analysis (H, M, cyclic, cumulative)")
    print("  Slots 8-13:  Soil Degradation (stiffness, strain, energy, entropy)")
    print("  Slots 14-19: Pile/Structural (geometry, EI, interaction, displacement)")
    print("  Slot 20:     FUSION (cross-attention over all 20)")
    print("  Output:      G/Gmax (PRIMARY), C_ratio, E_dissipated, Entropy")
    
    data = load_data()
    N = data['n_scenarios']
    T = data['n_steps']
    n_tr = int(0.85 * N)
    tr_idx = torch.arange(n_tr)
    te_idx = torch.arange(n_tr, N)
    
    model = Thermodynamic21SlotModel(n_steps=T)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {n_params:,}")
    print(f"Training: {EPOCHS} epochs ({n_tr} train / {len(te_idx)} test)\n")
    
    # Training loop
    losses = []
    comp_hist = {'G': [], 'C': [], 'E': [], 'S': [], 'slot0': [], 'monoC': [], 'thermo': []}
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        epoch_loss = 0.0
        comp_sum = {'G': 0.0, 'C': 0.0, 'E': 0.0, 'S': 0.0, 'slot0': 0.0, 'monoC': 0.0, 'thermo': 0.0}
        nb = 0
        
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i+BATCH]
            
            G_p, C_p, E_p, S_p, G_slot0, trig = model(
                data['pile'][idx],
                data['soil'][idx],
                data['load'][idx],
                data['environment'][idx],
            )
            
            tgt = data['targets'][idx]
            G_t, C_t, E_t, S_t = tgt[:,:,0], tgt[:,:,1], tgt[:,:,2], tgt[:,:,3]

            L_G = F.mse_loss(G_p, G_t)
            L_C = F.mse_loss(C_p, C_t)
            L_E = F.mse_loss(E_p, E_t)
            L_S = F.mse_loss(S_p, S_t)

            # Slot 0 is explicitly supervised to predict G/Gmax
            L_slot0 = F.mse_loss(G_slot0, G_t)

            # Load increase -> stiffness decrease
            load_mag = torch.sqrt(data['load'][idx][:, :, 0].pow(2) + data['load'][idx][:, :, 1].pow(2))
            d_load = load_mag[:, 1:] - load_mag[:, :-1]
            d_C = C_p[:, 1:] - C_p[:, :-1]
            inc_mask = (d_load > 0).float()
            L_mono_C = (F.relu(d_C) * inc_mask).sum() / (inc_mask.sum() + 1e-6)

            # Triggered condition: once triggered, stiffness should keep dropping
            L_trigger_C = (trig[:, 1:] * F.relu(d_C)).mean()

            # Thermodynamics: E and S should be non-decreasing in time
            d_E = E_p[:, 1:] - E_p[:, :-1]
            d_S = S_p[:, 1:] - S_p[:, :-1]
            L_thermo_mono = F.relu(-d_E).mean() + F.relu(-d_S).mean()

            # Thermodynamic consistency links
            L_E_cons = F.mse_loss(E_p, (1.0 - G_p).clamp(0.0, 1.0))
            L_S_cons = F.mse_loss(S_p, -torch.log(G_p.clamp(min=1e-3)))
            L_thermo = L_thermo_mono + 0.2 * L_E_cons + 0.1 * L_S_cons
            
            # Loss: heavy weight on G/Gmax (primary output)
            loss = (2.2 * L_G + 0.6 * L_C + 0.3 * L_E + 0.2 * L_S +
                    0.8 * L_slot0 + 0.8 * L_mono_C + 0.6 * L_trigger_C +
                    0.5 * L_thermo)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            comp_sum['G'] += L_G.item()
            comp_sum['C'] += L_C.item()
            comp_sum['E'] += L_E.item()
            comp_sum['S'] += L_S.item()
            comp_sum['slot0'] += L_slot0.item()
            comp_sum['monoC'] += (L_mono_C + L_trigger_C).item()
            comp_sum['thermo'] += L_thermo.item()
            nb += 1
        
        sched.step()
        losses.append(epoch_loss / nb)
        for k in comp_hist:
            comp_hist[k].append(comp_sum[k] / nb)
        
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  Ep {ep+1:3d}/{EPOCHS}  Loss: {losses[-1]:.6f}")
    
    # Evaluation
    print(f"\n{'-'*70}")
    print("TEST RESULTS")
    print(f"{'-'*70}")
    model.eval()
    with torch.no_grad():
        G_p, C_p, E_p, S_p, G_slot0, _ = model(
            data['pile'][te_idx],
            data['soil'][te_idx],
            data['load'][te_idx],
            data['environment'][te_idx],
        )
        
        tgt = data['targets'][te_idx]
        G_t, C_t, E_t, S_t = tgt[:,:,0], tgt[:,:,1], tgt[:,:,2], tgt[:,:,3]
        
        for nm, pred, true in [("G/Gmax", G_p, G_t),
                                ("C_ratio", C_p, C_t),
                                ("E_diss", E_p, E_t),
                                ("Entropy", S_p, S_t)]:
            mse = F.mse_loss(pred, true).item()
            print(f"  {nm:12s}  MSE={mse:.6f}")
        print(f"  {'Slot0-Gmax':12s}  MSE={F.mse_loss(G_slot0, G_t).item():.6f}")

    print_results_legend()

    plot_path = visualize_results(model, data, te_idx, losses, comp_hist)
    print(f"\nSaved graphs -> {plot_path}")
    
    print(f"\n{'='*70}\nDone!\n{'='*70}")
    return model


if __name__ == '__main__':
    model = train()
