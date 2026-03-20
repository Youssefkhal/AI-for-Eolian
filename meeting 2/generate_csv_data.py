"""
=============================================================================
Meeting 2 — CSV Data Generator for Specialized-Slot Transformer
=============================================================================
Purpose:
  Generate realistic synthetic data that mimics how an offshore wind
  turbine (OWT) monopile behaves during and after a typhoon.  The data
  is split into 4 domain CSV files — one for each physical aspect —
  plus a ground-truth file with the quantities the model must predict.

Generated files (saved to ./data/):
  1. structural.csv  — Pile geometry & material  (constant per scenario)
  2. soil.csv        — Soil properties & cyclic degradation (evolves)
  3. loading.csv     — Applied forces ramp up during the storm (evolves)
  4. environment.csv — Marine conditions intensify then scour deepens
  5. ground_truth.csv — True outputs: displacement u, stiffness K,
                        soil modulus ratio G/Gmax, frequency ratio f/f0

Each file has 200 scenarios × 30 time steps = 6 000 rows.
All files share the same scenario_id & step columns.

Physics models used:
  - Ramberg & Osgood (1943): load-displacement curve shape
  - Hardin & Drnevich (1972): soil shear-modulus degradation G/Gmax
  - Bhattacharya (2019): OWT natural frequency shift f/f0
  - DNV-OS-J101 / API RP 2GEO: pile-soil stiffness interaction
  - Equilibrium scour depth ≈ 1.3 × pile diameter (Sumer & Fredsøe)
=============================================================================
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# ── Configuration ───────────────────────────────────────────────────────────
N_SCENARIOS = 200        # number of loading scenarios
N_STEPS     = 30         # time steps per scenario
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))


def generate():
    """Generate all 4 domain CSV files + ground truth CSV."""

    print("=" * 60)
    print("Generating CSV data for Specialized-Slot Transformer")
    print("=" * 60)

    ids   = np.repeat(np.arange(N_SCENARIOS), N_STEPS)       # scenario_id
    steps = np.tile(np.arange(N_STEPS), N_SCENARIOS)          # step index
    t_norm = np.tile(np.linspace(0, 1, N_STEPS), N_SCENARIOS) # normalized time

    # ====================================================================
    # 1. STRUCTURAL — pile geometry & material (constant per scenario)
    # ====================================================================
    L_pile  = np.random.uniform(20, 40, N_SCENARIOS)     # embedded length [m]
    D_pile  = np.random.uniform(4, 8, N_SCENARIOS)       # outer diameter [m]
    t_wall  = np.random.uniform(0.05, 0.10, N_SCENARIOS) # wall thickness [m]
    EI      = np.random.uniform(1e10, 5e10, N_SCENARIOS) # flexural rigidity [N·m²]

    # Broadcast to all steps (constant per scenario)
    L_pile_full = np.repeat(L_pile, N_STEPS)
    D_pile_full = np.repeat(D_pile, N_STEPS)
    t_wall_full = np.repeat(t_wall, N_STEPS)
    EI_full     = np.repeat(EI, N_STEPS)

    df_struct = pd.DataFrame({
        'scenario_id': ids,
        'step':        steps,
        'L_pile_m':    np.round(L_pile_full, 3),
        'D_pile_m':    np.round(D_pile_full, 3),
        't_wall_m':    np.round(t_wall_full, 4),
        'EI_Nm2':      np.round(EI_full, 0),
    })

    # ====================================================================
    # 2. SOIL — properties & evolving state
    # ====================================================================
    Su_init  = np.random.uniform(30, 120, N_SCENARIOS)    # undrained shear strength [kPa]
    G0       = np.random.uniform(20e3, 80e3, N_SCENARIOS) # initial shear modulus [kPa]
    e_void   = np.random.uniform(0.5, 1.2, N_SCENARIOS)   # void ratio
    OCR      = np.random.uniform(1.0, 8.0, N_SCENARIOS)   # over-consolidation ratio

    # Su degrades slightly with cycling (pore-pressure build-up)
    Su_full  = np.repeat(Su_init, N_STEPS)
    deg_factor = 1.0 - 0.15 * t_norm  # up to 15% degradation
    Su_full  = Su_full * deg_factor + np.random.normal(0, 0.5, len(Su_full))

    G0_full  = np.repeat(G0, N_STEPS)
    e_full   = np.repeat(e_void, N_STEPS)
    OCR_full = np.repeat(OCR, N_STEPS)

    df_soil = pd.DataFrame({
        'scenario_id':  ids,
        'step':         steps,
        'Su_kPa':       np.round(Su_full, 2),
        'G0_kPa':       np.round(G0_full, 1),
        'void_ratio':   np.round(e_full, 3),
        'OCR':          np.round(OCR_full, 2),
    })

    # ====================================================================
    # 3. LOADING — applied forces & cycle information
    # ====================================================================
    F_max    = np.random.uniform(1e3, 5e3, N_SCENARIOS)   # max lateral force [kN]
    M_ratio  = np.random.uniform(5, 15, N_SCENARIOS)      # moment arm [m]
    freq     = np.random.uniform(0.1, 0.35, N_SCENARIOS)  # loading frequency [Hz]
    N_cyc_total = np.random.uniform(100, 1e4, N_SCENARIOS) # total target cycles

    F_lat_full = np.repeat(F_max, N_STEPS) * t_norm       # ramp up force
    F_lat_full += np.random.normal(0, 30, len(F_lat_full)) # small noise
    F_lat_full = np.clip(F_lat_full, 0, None)

    M_base_full = F_lat_full * np.repeat(M_ratio, N_STEPS) # base moment = F × arm

    # Cumulative cycles = fraction × total
    N_cyc_full = np.repeat(N_cyc_total, N_STEPS) * t_norm
    freq_full  = np.repeat(freq, N_STEPS)

    df_loading = pd.DataFrame({
        'scenario_id':    ids,
        'step':           steps,
        'F_lateral_kN':   np.round(F_lat_full, 2),
        'M_base_kNm':     np.round(M_base_full, 1),
        'N_cycles':       np.round(N_cyc_full, 0).astype(int),
        'freq_Hz':        np.round(freq_full, 3),
    })

    # ====================================================================
    # 4. ENVIRONMENT — marine & metocean conditions
    # ====================================================================
    wave_H_base   = np.random.uniform(1, 8, N_SCENARIOS)     # sig. wave height [m]
    wind_V_base   = np.random.uniform(5, 25, N_SCENARIOS)     # wind speed [m/s]
    water_depth   = np.random.uniform(10, 40, N_SCENARIOS)     # water depth [m]
    scour_init    = np.random.uniform(0.0, 1.5, N_SCENARIOS)  # initial scour [m]

    # Waves & wind intensify during loading (typhoon ramp)
    wave_H_full = np.repeat(wave_H_base, N_STEPS) * (0.5 + 0.7 * t_norm)
    wave_H_full += np.random.normal(0, 0.1, len(wave_H_full))
    wave_H_full = np.clip(wave_H_full, 0.1, None)

    wind_V_full = np.repeat(wind_V_base, N_STEPS) * (0.5 + 0.7 * t_norm)
    wind_V_full += np.random.normal(0, 0.3, len(wind_V_full))
    wind_V_full = np.clip(wind_V_full, 1, None)

    depth_full  = np.repeat(water_depth, N_STEPS)

    # Scour deepens over time (equilibrium scour ~ 1.3D)
    scour_eq    = 1.3 * D_pile
    scour_full  = np.repeat(scour_init, N_STEPS) + \
                  np.repeat(scour_eq - scour_init, N_STEPS) * (1 - np.exp(-3 * t_norm))
    scour_full += np.random.normal(0, 0.02, len(scour_full))
    scour_full  = np.clip(scour_full, 0, None)

    df_env = pd.DataFrame({
        'scenario_id':    ids,
        'step':           steps,
        'wave_H_m':       np.round(wave_H_full, 3),
        'wind_V_ms':      np.round(wind_V_full, 2),
        'water_depth_m':  np.round(depth_full, 2),
        'scour_depth_m':  np.round(scour_full, 3),
    })

    # ====================================================================
    # 5. GROUND TRUTH — physical outputs the model must predict
    # ====================================================================
    # We compute 4 output quantities from first-principles formulas.
    # These are the "answers" the transformer must learn to predict.

    # --- Normalize inputs to [0,1] for physics formulas ---
    L_n  = L_pile_full / 40.0   # normalized embedded length
    D_n  = D_pile_full / 8.0    # normalized diameter
    F_n  = F_lat_full / 5e3     # normalized lateral force

    # (a) Initial lateral stiffness K0 [proportional to soil modulus × pile size]
    #     Bigger pile + deeper embedment + stiffer soil → higher K0
    K0 = (2.0 * D_n + 1.5 * L_n + 0.5) * G0_full / 50e3
    K0 += np.random.normal(0, 0.01, len(K0)) * np.abs(K0)  # 1% noise

    # (b) Yield force F_y [force at which behavior turns nonlinear]
    #     Larger diameter + deeper pile + stronger soil → higher yield
    F_y = (0.5 * D_n + 0.3 * L_n + 0.2) * Su_full / 75.0
    F_y = np.clip(F_y, 0.05, None)

    # Smoothness exponent (varies per scenario) for Ramberg-Osgood curve
    n_smooth = 2.0 + 0.5 * np.repeat(np.random.rand(N_SCENARIOS), N_STEPS)

    # (c) Displacement u — Ramberg-Osgood (1943) model
    #     u = (F/K0) × [1 + (F/F_y)^n]^(1/n)
    #     Below yield: u ≈ F/K0 (linear elastic)
    #     Above yield: u grows faster (plastic/nonlinear)
    ratio = np.clip(F_n / (F_y + 1e-8), 0, None)
    u = (F_n / (K0 + 1e-8)) * (1.0 + ratio ** n_smooth) ** (1.0 / n_smooth)
    u += np.random.normal(0, 0.003, len(u)) * np.abs(u)
    u = np.clip(u, 0, None)

    # (d) G/Gmax — Hardin & Drnevich (1972) degradation
    #     As strain increases, soil shear modulus drops:
    #     G/Gmax = 1 / (1 + γ^1.2)   where γ = strain
    #     Value of 1.0 = undisturbed soil; value → 0 = fully degraded
    strain = u / (L_n * 0.5 + 1e-8)
    G_ratio = 1.0 / (1.0 + np.clip(strain, 0, None) ** 1.2)

    # (e) Tangent stiffness K = K0 × G/Gmax
    #     Stiffness drops as soil degrades
    K = K0 * G_ratio

    # (f) Natural frequency ratio f/f0 — Bhattacharya (2019)
    #     Two effects reduce the monopile's natural frequency:
    #       1. Soil degradation (G/Gmax drops → foundation softens)
    #       2. Scour deepening  (effective embedded length shrinks)
    #     f/f0 = sqrt(G/Gmax) × (L_eff / L)^0.3
    L_eff = L_n - scour_full / 40.0   # effective length shrinks with scour
    L_eff = np.clip(L_eff, 0.1, None)
    f_ratio = np.sqrt(G_ratio) * (L_eff / L_n) ** 0.3
    f_ratio = np.clip(f_ratio, 0.3, 1.0)
    f_ratio += np.random.normal(0, 0.005, len(f_ratio))  # small noise
    f_ratio = np.clip(f_ratio, 0.3, 1.0)

    df_gt = pd.DataFrame({
        'scenario_id':     ids,
        'step':            steps,
        'u_displacement':  np.round(u, 6),
        'K_stiffness':     np.round(K, 6),
        'G_ratio':         np.round(G_ratio, 6),
        'f_over_f0':       np.round(f_ratio, 6),
    })

    # ====================================================================
    # SAVE
    # ====================================================================
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    files = {
        'structural.csv':  df_struct,
        'soil.csv':        df_soil,
        'loading.csv':     df_loading,
        'environment.csv': df_env,
        'ground_truth.csv': df_gt,
    }

    for fname, df in files.items():
        path = os.path.join(data_dir, fname)
        df.to_csv(path, index=False)
        print(f"  ✓ {fname:22s}  →  {len(df):,} rows  ({df.shape[1]} cols)")

    print(f"\nAll files saved to: {data_dir}")
    print(f"Scenarios: {N_SCENARIOS},  Steps/scenario: {N_STEPS}")
    print(f"Total rows per file: {N_SCENARIOS * N_STEPS:,}")
    print("=" * 60)

    return data_dir


if __name__ == '__main__':
    generate()
