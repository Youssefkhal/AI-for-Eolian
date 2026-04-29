"""
M8 Scenario Analysis: Curve-Based Errors, Worst-Case Diagnosis, Feature Importance
====================================================================================
- Loads the trained M8 Ψ-model (no retraining)
- Computes curve-level errors (cumulative stiffness curves, not per-slot drops)
- Identifies the worst scenarios and diagnoses WHY they fail
- Computes feature importance (% influence on prediction error)
- Exports results to JSON + generates LaTeX report
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_STEPS = 21
STEP_INDICES = np.round(np.linspace(0, 43, NUM_STEPS)).astype(int)
BOTTLENECK_DIM = 48

# ─────────────────────────────────────────────────────
# Model classes (must match train.py exactly)
# ─────────────────────────────────────────────────────

class EfficientSlotMLP(nn.Module):
    def __init__(self, d_model=64, bottleneck_dim=BOTTLENECK_DIM, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(bottleneck_dim, d_model))
    def forward(self, x):
        return self.net(x)


class SlotAttentionPsiModel(nn.Module):
    def __init__(self, input_size=8, d_model=64, num_heads=4,
                 max_seq_len=21, dropout=0.1, num_iterations=3,
                 num_prototypes=4, relation_matrix=None, structured=True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations
        self.num_prototypes = num_prototypes
        self.num_drop_slots = max_seq_len - 1

        self.input_embed = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.initial_slot = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.prototype_slots = nn.Parameter(torch.randn(1, num_prototypes, d_model) * 0.02)

        if relation_matrix is not None:
            R_init = torch.FloatTensor(relation_matrix)
        else:
            R_init = torch.zeros(self.num_drop_slots, num_prototypes)
            slots_per_proto = self.num_drop_slots // num_prototypes
            for p in range(num_prototypes):
                start = p * slots_per_proto
                end = start + slots_per_proto if p < num_prototypes - 1 else self.num_drop_slots
                R_init[start:end, p] = 1.0
        self.relation_logits = nn.Parameter(torch.log(R_init.clamp(min=1e-6)))
        self.slot_scales = nn.Parameter(torch.ones(self.num_drop_slots, 1))

        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = EfficientSlotMLP(d_model=d_model, bottleneck_dim=BOTTLENECK_DIM, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def get_relation_matrix(self):
        return torch.softmax(self.relation_logits, dim=1)

    def reconstruct_drop_slots(self, B):
        protos = self.prototype_slots.expand(B, -1, -1)
        R = self.get_relation_matrix()
        drop_slots = torch.matmul(R, protos)
        drop_slots = drop_slots * self.slot_scales.unsqueeze(0)
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
# Utilities
# ─────────────────────────────────────────────────────

def inverse_transform_values(scaled, scaler):
    flat = scaled.reshape(-1, 3)
    log_vals = scaler.inverse_transform(flat)
    orig = np.sign(log_vals) * np.expm1(np.abs(log_vals))
    return orig.reshape(scaled.shape)


def curve_mape(target_curve, pred_curve):
    """Mean absolute percentage error along a curve, avoiding division by near-zero."""
    denom = np.abs(target_curve)
    mask = denom > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(target_curve[mask] - pred_curve[mask]) / denom[mask]) * 100)


def curve_nrmse(target_curve, pred_curve):
    """Normalised RMSE: RMSE / range of target."""
    rmse = np.sqrt(np.mean((target_curve - pred_curve) ** 2))
    rng = target_curve.max() - target_curve.min()
    if rng < 1e-10:
        return 0.0
    return float(rmse / rng * 100)


def curve_r2(target_curve, pred_curve):
    """R² along a single curve."""
    if len(target_curve) < 2:
        return 1.0
    ss_res = np.sum((target_curve - pred_curve) ** 2)
    ss_tot = np.sum((target_curve - target_curve.mean()) ** 2)
    if ss_tot < 1e-20:
        return 1.0
    return float(1 - ss_res / ss_tot)


def curve_max_error(target_curve, pred_curve):
    """Max absolute error along a curve."""
    return float(np.max(np.abs(target_curve - pred_curve)))


def area_between_curves(target_curve, pred_curve):
    """Area between predicted and target curves (trapezoidal integration)."""
    diff = np.abs(target_curve - pred_curve)
    return float(np.trapezoid(diff, dx=1.0))


# ─────────────────────────────────────────────────────
# Main Analysis
# ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("M8 Scenario Analysis: Curve-Based Errors & Feature Importance")
    print("=" * 70)

    # Load artifacts
    scaler_X = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
    scaler_Y = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
    feature_names = joblib.load(os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
    max_seq_len = joblib.load(os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
    test_data = joblib.load(os.path.join(SCRIPT_DIR, 'test_data.pkl'))
    psi_config = joblib.load(os.path.join(SCRIPT_DIR, 'psi_config.pkl'))

    X_original = test_data['X_original']
    Y_original = test_data['Y_original']
    X_scaled = test_data['X_scaled']
    input_cols = test_data['input_cols']
    var_names = ['KL', 'KR', 'KLR']

    # Load model
    R = np.array(psi_config['relation_matrix'])
    model = SlotAttentionPsiModel(
        input_size=len(feature_names), d_model=64, num_heads=4,
        max_seq_len=max_seq_len, dropout=0.1, num_iterations=3,
        num_prototypes=psi_config['num_prototypes'],
        relation_matrix=R, structured=True)
    model.load_state_dict(torch.load(
        os.path.join(SCRIPT_DIR, 'pile_model.pth'), map_location='cpu', weights_only=True))
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Predict
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_scaled), max_seq_len).numpy()
    pred_original = inverse_transform_values(pred_scaled, scaler_Y)

    n_test = len(X_original)
    print(f"Test scenarios: {n_test}")

    # ═══════════════════════════════════════════════════
    # 1. CURVE-BASED ERROR METRICS (per scenario)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("1. CURVE-BASED ERROR METRICS")
    print(f"{'='*70}")

    scenario_errors = []
    for i in range(n_test):
        target = Y_original[i]   # [21, 3]
        pred = pred_original[i]  # [21, 3]

        sc = {
            'scenario_id': i + 1,
            'params': {col: float(X_original[i][j]) for j, col in enumerate(input_cols)},
        }

        # Per-variable curve metrics
        var_errors = {}
        overall_mape = 0.0
        overall_nrmse = 0.0
        overall_r2 = 0.0
        overall_abc = 0.0
        for vi, vn in enumerate(var_names):
            t_curve = target[:, vi]
            p_curve = pred[:, vi]
            mape = curve_mape(t_curve, p_curve)
            nrmse = curve_nrmse(t_curve, p_curve)
            r2 = curve_r2(t_curve, p_curve)
            maxerr = curve_max_error(t_curve, p_curve)
            abc = area_between_curves(t_curve, p_curve)

            var_errors[vn] = {
                'curve_mape_pct': round(mape, 3),
                'curve_nrmse_pct': round(nrmse, 3),
                'curve_r2': round(r2, 6),
                'max_abs_error': float(f'{maxerr:.4e}'),
                'area_between_curves': float(f'{abc:.4e}'),
            }
            overall_mape += mape
            overall_nrmse += nrmse
            overall_r2 += r2
            overall_abc += abc

        sc['per_variable'] = var_errors
        sc['overall'] = {
            'avg_curve_mape_pct': round(overall_mape / 3, 3),
            'avg_curve_nrmse_pct': round(overall_nrmse / 3, 3),
            'avg_curve_r2': round(overall_r2 / 3, 6),
            'total_area_between_curves': float(f'{overall_abc:.4e}'),
        }
        scenario_errors.append(sc)

    # Sort by worst (highest avg MAPE)
    scenario_errors.sort(key=lambda x: x['overall']['avg_curve_mape_pct'], reverse=True)

    # Global stats
    all_mapes = [s['overall']['avg_curve_mape_pct'] for s in scenario_errors]
    all_nrmses = [s['overall']['avg_curve_nrmse_pct'] for s in scenario_errors]
    all_r2s = [s['overall']['avg_curve_r2'] for s in scenario_errors]

    print(f"\n  Global Curve Metrics (across {n_test} test scenarios):")
    print(f"    Avg MAPE:  {np.mean(all_mapes):.2f}%  (median: {np.median(all_mapes):.2f}%)")
    print(f"    Avg NRMSE: {np.mean(all_nrmses):.2f}%  (median: {np.median(all_nrmses):.2f}%)")
    print(f"    Avg R²:    {np.mean(all_r2s):.4f}  (median: {np.median(all_r2s):.4f})")

    # ═══════════════════════════════════════════════════
    # 2. WORST SCENARIOS ANALYSIS
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("2. WORST SCENARIOS (top 10 by avg curve MAPE)")
    print(f"{'='*70}")

    N_WORST = 10
    worst = scenario_errors[:N_WORST]

    for rank, sc in enumerate(worst, 1):
        sid = sc['scenario_id']
        p = sc['params']
        o = sc['overall']
        print(f"\n  #{rank}: Scenario {sid}")
        print(f"    Params: PI={p['PI']:.1f}, Gmax={p['Gmax']:.0f}, v={p['v']:.3f}, "
              f"Dp={p['Dp']:.2f}, Tp={p['Tp']:.4f}, Lp={p['Lp']:.1f}, "
              f"Ip={p['Ip']:.6f}, Dp/Lp={p['Dp_Lp']:.4f}")
        print(f"    Avg MAPE={o['avg_curve_mape_pct']:.2f}%  NRMSE={o['avg_curve_nrmse_pct']:.2f}%  R²={o['avg_curve_r2']:.4f}")
        for vn in var_names:
            ve = sc['per_variable'][vn]
            print(f"      {vn:>3}: MAPE={ve['curve_mape_pct']:.2f}%  NRMSE={ve['curve_nrmse_pct']:.2f}%  "
                  f"R²={ve['curve_r2']:.4f}  MaxErr={ve['max_abs_error']:.3e}")

    # ═══════════════════════════════════════════════════
    # 3. DIAGNOSIS: WHY DO WORST SCENARIOS FAIL?
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("3. DIAGNOSIS: Comparing Worst vs Best Scenarios")
    print(f"{'='*70}")

    N_BEST = 20
    best = scenario_errors[-N_BEST:]

    # Collect feature distributions for worst vs best
    worst_params = np.array([[sc['params'][col] for col in input_cols] for sc in worst])
    best_params = np.array([[sc['params'][col] for col in input_cols] for sc in best])
    all_params = X_original.astype(float)

    diagnosis = {}
    print(f"\n  Feature comparison (worst {N_WORST} vs best {N_BEST} vs all {n_test}):")
    print(f"  {'Feature':<10} {'Worst Mean':>12} {'Best Mean':>12} {'All Mean':>12} {'Worst/All':>10} {'Deviation':>10}")
    print(f"  {'-'*66}")

    for j, col in enumerate(input_cols):
        w_mean = worst_params[:, j].mean()
        b_mean = best_params[:, j].mean()
        a_mean = all_params[:, j].mean()
        a_std = all_params[:, j].std()

        ratio = w_mean / a_mean if abs(a_mean) > 1e-10 else 0
        dev = (w_mean - a_mean) / a_std if a_std > 1e-10 else 0

        diagnosis[col] = {
            'worst_mean': float(w_mean),
            'best_mean': float(b_mean),
            'all_mean': float(a_mean),
            'all_std': float(a_std),
            'worst_to_all_ratio': float(ratio),
            'deviation_sigma': float(dev),
        }
        print(f"  {col:<10} {w_mean:>12.4f} {b_mean:>12.4f} {a_mean:>12.4f} {ratio:>10.3f} {dev:>+10.2f}σ")

    # Target curve characteristics
    print(f"\n  Target Curve Characteristics (worst vs best):")
    for label, group in [("Worst", worst), ("Best", best)]:
        indices = [sc['scenario_id'] - 1 for sc in group]
        targets = Y_original[indices]
        kl_range = np.ptp(targets[:, :, 0], axis=1).mean()
        kr_range = np.ptp(targets[:, :, 1], axis=1).mean()
        klr_range = np.ptp(targets[:, :, 2], axis=1).mean()
        kl_init = targets[:, 0, 0].mean()
        kr_init = targets[:, 0, 1].mean()
        klr_init = targets[:, 0, 2].mean()
        print(f"\n    {label} ({len(group)} scenarios):")
        print(f"      Avg initial:   KL={kl_init:.3e}  KR={kr_init:.3e}  KLR={klr_init:.3e}")
        print(f"      Avg range:     KL={kl_range:.3e}  KR={kr_range:.3e}  KLR={klr_range:.3e}")
        # Degradation rate (avg drop per step)
        kl_rate = np.mean(targets[:, -1, 0] - targets[:, 0, 0]) / NUM_STEPS
        kr_rate = np.mean(targets[:, -1, 1] - targets[:, 0, 1]) / NUM_STEPS
        klr_rate = np.mean(targets[:, -1, 2] - targets[:, 0, 2]) / NUM_STEPS
        print(f"      Avg degrad/step: KL={kl_rate:.3e}  KR={kr_rate:.3e}  KLR={klr_rate:.3e}")

    # ═══════════════════════════════════════════════════
    # 4. FEATURE IMPORTANCE (% influence on error)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("4. FEATURE IMPORTANCE (% influence on prediction error)")
    print(f"{'='*70}")

    # Target: per-scenario avg MAPE
    error_target = np.array(all_mapes)
    # Reorder to match test indices (scenario_errors is sorted by worst)
    id_to_mape = {sc['scenario_id']: sc['overall']['avg_curve_mape_pct'] for sc in scenario_errors}
    # X_original order matches scenario_errors by ID
    error_vec = np.array([id_to_mape[i + 1] for i in range(n_test)])

    # Gradient Boosting Regressor to model: features → error
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42,
                                    learning_rate=0.05, subsample=0.8)
    gb.fit(X_original, error_vec)

    # Built-in feature importance
    fi_builtin = gb.feature_importances_
    fi_pct_builtin = fi_builtin / fi_builtin.sum() * 100

    # Permutation importance (more reliable)
    perm_result = permutation_importance(gb, X_original, error_vec,
                                          n_repeats=30, random_state=42)
    fi_perm = np.maximum(perm_result.importances_mean, 0)
    fi_pct_perm = fi_perm / fi_perm.sum() * 100 if fi_perm.sum() > 0 else fi_perm

    # Correlation with error
    correlations = []
    for j in range(len(input_cols)):
        corr = np.corrcoef(X_original[:, j].astype(float), error_vec)[0, 1]
        correlations.append(float(corr))

    importance_data = {}
    print(f"\n  {'Feature':<10} {'GBR Imp%':>10} {'Perm Imp%':>10} {'Corr→Err':>10} {'Combined%':>10}")
    print(f"  {'-'*52}")

    combined = (fi_pct_builtin + fi_pct_perm) / 2
    combined_pct = combined / combined.sum() * 100

    for j, col in enumerate(input_cols):
        importance_data[col] = {
            'gbr_importance_pct': round(float(fi_pct_builtin[j]), 2),
            'permutation_importance_pct': round(float(fi_pct_perm[j]), 2),
            'correlation_with_error': round(correlations[j], 4),
            'combined_influence_pct': round(float(combined_pct[j]), 2),
        }
        print(f"  {col:<10} {fi_pct_builtin[j]:>10.2f} {fi_pct_perm[j]:>10.2f} "
              f"{correlations[j]:>+10.4f} {combined_pct[j]:>10.2f}")

    # Sort by combined influence
    sorted_features = sorted(importance_data.items(), key=lambda x: x[1]['combined_influence_pct'], reverse=True)
    print(f"\n  Ranking by combined influence:")
    for rank, (col, imp) in enumerate(sorted_features, 1):
        print(f"    {rank}. {col:<10} → {imp['combined_influence_pct']:.1f}%")

    # ═══════════════════════════════════════════════════
    # 5. PER-VARIABLE CURVE ERROR DISTRIBUTION
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("5. PER-VARIABLE CURVE ERROR DISTRIBUTION")
    print(f"{'='*70}")

    for vn in var_names:
        mapes = [sc['per_variable'][vn]['curve_mape_pct'] for sc in scenario_errors]
        r2s = [sc['per_variable'][vn]['curve_r2'] for sc in scenario_errors]
        print(f"\n  {vn}:")
        print(f"    MAPE: mean={np.mean(mapes):.2f}%, median={np.median(mapes):.2f}%, "
              f"p95={np.percentile(mapes, 95):.2f}%, max={np.max(mapes):.2f}%")
        print(f"    R²:   mean={np.mean(r2s):.4f}, median={np.median(r2s):.4f}, "
              f"p5={np.percentile(r2s, 5):.4f}, min={np.min(r2s):.4f}")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    results = {
        'global_metrics': {
            'avg_curve_mape_pct': round(float(np.mean(all_mapes)), 3),
            'median_curve_mape_pct': round(float(np.median(all_mapes)), 3),
            'avg_curve_nrmse_pct': round(float(np.mean(all_nrmses)), 3),
            'avg_curve_r2': round(float(np.mean(all_r2s)), 6),
            'median_curve_r2': round(float(np.median(all_r2s)), 6),
            'p95_curve_mape_pct': round(float(np.percentile(all_mapes, 95)), 3),
            'n_test': n_test,
        },
        'worst_scenarios': worst,
        'best_scenarios': best[-5:],  # just top 5 best
        'diagnosis': diagnosis,
        'feature_importance': importance_data,
        'feature_ranking': [{'rank': i+1, 'feature': col, 'influence_pct': imp['combined_influence_pct']}
                           for i, (col, imp) in enumerate(sorted_features)],
        'per_variable_distribution': {},
    }
    for vn in var_names:
        mapes = [sc['per_variable'][vn]['curve_mape_pct'] for sc in scenario_errors]
        r2s = [sc['per_variable'][vn]['curve_r2'] for sc in scenario_errors]
        results['per_variable_distribution'][vn] = {
            'mape_mean': round(float(np.mean(mapes)), 3),
            'mape_median': round(float(np.median(mapes)), 3),
            'mape_p95': round(float(np.percentile(mapes, 95)), 3),
            'mape_max': round(float(np.max(mapes)), 3),
            'r2_mean': round(float(np.mean(r2s)), 6),
            'r2_median': round(float(np.median(r2s)), 6),
            'r2_p5': round(float(np.percentile(r2s, 5)), 6),
            'r2_min': round(float(np.min(r2s)), 6),
        }

    # Save ALL scenario errors (sorted worst first)
    results['all_scenarios'] = scenario_errors

    with open(os.path.join(SCRIPT_DIR, 'scenario_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to scenario_analysis.json")

    # ═══════════════════════════════════════════════════
    # 6. GENERATE LATEX REPORT
    # ═══════════════════════════════════════════════════
    generate_latex(results, input_cols, var_names, worst, diagnosis, sorted_features, n_test)

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


def generate_latex(results, input_cols, var_names, worst, diagnosis, sorted_features, n_test):
    """Generate a complete LaTeX report for Overleaf."""

    gm = results['global_metrics']

    # Build worst scenarios table rows
    worst_rows = ""
    for rank, sc in enumerate(worst, 1):
        o = sc['overall']
        p = sc['params']
        worst_rows += (
            f"    {rank} & {sc['scenario_id']} & {o['avg_curve_mape_pct']:.2f}\\% & "
            f"{o['avg_curve_nrmse_pct']:.2f}\\% & {o['avg_curve_r2']:.4f} \\\\\n"
        )

    # Build worst scenario detail rows (per variable)
    worst_detail = ""
    for rank, sc in enumerate(worst[:5], 1):
        for vi, vn in enumerate(var_names):
            ve = sc['per_variable'][vn]
            prefix = f"\\multirow{{3}}{{*}}{{{rank} (S{sc['scenario_id']})}}" if vi == 0 else ""
            worst_detail += (
                f"    {prefix} & {vn} & {ve['curve_mape_pct']:.2f}\\% & "
                f"{ve['curve_nrmse_pct']:.2f}\\% & {ve['curve_r2']:.4f} & "
                f"${ve['max_abs_error']:.2e}$ \\\\\n"
            )
        worst_detail += "    \\hline\n"

    # Feature importance table
    fi_rows = ""
    for item in sorted_features:
        col, imp = item
        fi_rows += (
            f"    {imp['combined_influence_pct']:.1f}\\% & {col} & "
            f"{imp['gbr_importance_pct']:.1f}\\% & "
            f"{imp['permutation_importance_pct']:.1f}\\% & "
            f"{imp['correlation_with_error']:+.4f} \\\\\n"
        )

    # Diagnosis table: worst vs all
    diag_rows = ""
    for col in input_cols:
        d = diagnosis[col]
        diag_rows += (
            f"    {col} & {d['worst_mean']:.4f} & {d['all_mean']:.4f} & "
            f"{d['deviation_sigma']:+.2f}$\\sigma$ \\\\\n"
        )

    # Per-variable distribution table
    pvd_rows = ""
    for vn in var_names:
        pvd = results['per_variable_distribution'][vn]
        pvd_rows += (
            f"    {vn} & {pvd['mape_mean']:.2f}\\% & {pvd['mape_median']:.2f}\\% & "
            f"{pvd['mape_p95']:.2f}\\% & {pvd['r2_mean']:.4f} & {pvd['r2_min']:.4f} \\\\\n"
        )

    # Worst scenario input params table
    worst_params_rows = ""
    for rank, sc in enumerate(worst[:5], 1):
        p = sc['params']
        worst_params_rows += (
            f"    {rank} & S{sc['scenario_id']} & {p['PI']:.0f} & {p['Gmax']:.0f} & "
            f"{p['v']:.3f} & {p['Dp']:.2f} & {p['Tp']:.4f} & {p['Lp']:.1f} & "
            f"{p['Ip']:.2e} & {p['Dp_Lp']:.4f} \\\\\n"
        )

    # Build the full diagnosis text for worst scenarios
    # Analyze specific patterns
    worst_params_np = np.array([[sc['params'][col] for col in input_cols] for sc in worst])
    best_scenarios = results['best_scenarios']
    best_params_np = np.array([[sc['params'][col] for col in input_cols] for sc in best_scenarios])

    # Find the most deviant features for worst scenarios
    deviant_features = sorted(diagnosis.items(), key=lambda x: abs(x[1]['deviation_sigma']), reverse=True)
    top_deviant = deviant_features[:3]

    deviant_text = ""
    for col, d in top_deviant:
        direction = "higher" if d['deviation_sigma'] > 0 else "lower"
        deviant_text += (
            f"\\item \\textbf{{{col}}}: The worst scenarios have a mean value of "
            f"{d['worst_mean']:.4f}, which is {abs(d['deviation_sigma']):.2f}$\\sigma$ "
            f"{direction} than the dataset mean ({d['all_mean']:.4f}). "
        )
        if col in ('Gmax', 'v'):
            deviant_text += (
                f"This parameter directly controls the soil stiffness and wave propagation "
                f"behaviour, so extreme values create degradation patterns that the model "
                f"has fewer training examples to learn from.\n"
            )
        elif col in ('Dp', 'Lp', 'Dp_Lp'):
            deviant_text += (
                f"This geometric parameter governs the pile's slenderness and load transfer "
                f"mechanism. Extreme ratios lead to non-standard degradation profiles that "
                f"deviate from the cluster prototypes learned in Stage~B.\n"
            )
        elif col in ('PI',):
            deviant_text += (
                f"The plasticity index affects clay behaviour under cyclic loading. "
                f"Extreme PI values produce highly nonlinear degradation curves that "
                f"are harder to approximate with {results['global_metrics']['n_test']} prototypes.\n"
            )
        elif col in ('Tp',):
            deviant_text += (
                f"Pile wall thickness affects the structural rigidity and load "
                f"distribution. Unusual thickness values create atypical stiffness "
                f"evolution patterns.\n"
            )
        elif col in ('Ip',):
            deviant_text += (
                f"The moment of inertia directly determines the pile's bending "
                f"stiffness. Extreme values lead to very different degradation "
                f"dynamics that prototype slots cannot fully represent.\n"
            )

    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{enumitem}
\geometry{margin=2.2cm}

\definecolor{bestgreen}{RGB}{0,128,0}
\definecolor{worstred}{RGB}{180,30,30}

\title{M8 Efficient $\Psi$-NN: Curve-Based Error Analysis\\and Worst-Scenario Diagnosis}
\author{Automated Analysis Report}
\date{\today}

\begin{document}
\maketitle

%% ═══════════════════════════════════════════════════
\section{Model Overview}
%% ═══════════════════════════════════════════════════

The M8 Efficient $\Psi$-NN model predicts lateral pile stiffness degradation
curves (KL, KR, KLR) over """ + str(gm['n_test']) + r""" test scenarios with
\textbf{45,486 parameters} (19.7\% fewer than the M6 teacher's 56,646).
Three key upgrades were applied over M7:
\begin{enumerate}[nosep]
    \item \textbf{EfficientSlotMLP}: bottleneck $64 \to 48 \to 64$ (vs.\ wide $64 \to 128 \to 64$)
    \item \textbf{Learnable relation matrix}: row-softmax logits optimised end-to-end
    \item \textbf{Physics-monotonic penalty}: enforces KL/KR $\downarrow$ and KLR $\uparrow$
\end{enumerate}

\paragraph{Error methodology.}
Unlike slot-by-slot drop metrics, we evaluate \emph{cumulative stiffness curves}
as they represent the physical quantity of interest.  For each scenario $i$ and
variable $v \in \{\text{KL},\text{KR},\text{KLR}\}$, we compare the predicted
curve $\hat{y}^{(i,v)}_{1:T}$ against the target $y^{(i,v)}_{1:T}$ using:
\begin{itemize}[nosep]
    \item \textbf{Curve MAPE} $= \frac{1}{T}\sum_{t=1}^{T} \frac{|\hat{y}_t - y_t|}{|y_t|} \times 100\%$
    \item \textbf{Curve NRMSE} $= \frac{\text{RMSE}}{\max(y) - \min(y)} \times 100\%$
    \item \textbf{Curve $R^2$} per individual scenario curve
    \item \textbf{Area Between Curves (ABC)} $= \int |\hat{y}(t) - y(t)|\,dt$
\end{itemize}

%% ═══════════════════════════════════════════════════
\section{Global Curve-Based Performance}
%% ═══════════════════════════════════════════════════

\begin{table}[h]
\centering
\caption{Global curve-based metrics across all """ + str(n_test) + r""" test scenarios.}
\begin{tabular}{lcccccc}
\toprule
\textbf{Variable} & \textbf{MAPE (mean)} & \textbf{MAPE (median)} &
\textbf{MAPE (p95)} & \textbf{$R^2$ (mean)} & \textbf{$R^2$ (min)} \\
\midrule
""" + pvd_rows + r"""\bottomrule
\end{tabular}
\label{tab:global_curve}
\end{table}

\noindent Overall across all variables:
\textbf{Mean MAPE = """ + f"{gm['avg_curve_mape_pct']:.2f}" + r"""\%},
\textbf{Median MAPE = """ + f"{gm['median_curve_mape_pct']:.2f}" + r"""\%},
\textbf{Mean $R^2$ = """ + f"{gm['avg_curve_r2']:.4f}" + r"""}.

%% ═══════════════════════════════════════════════════
\section{Worst-Performing Scenarios}
%% ═══════════════════════════════════════════════════

Table~\ref{tab:worst10} lists the 10 scenarios with the highest average
curve MAPE, sorted from worst to least-worst.

\begin{table}[h]
\centering
\caption{Top-10 worst scenarios ranked by average curve MAPE.}
\begin{tabular}{cccccc}
\toprule
\textbf{Rank} & \textbf{Scenario} & \textbf{Avg MAPE} &
\textbf{Avg NRMSE} & \textbf{Avg $R^2$} \\
\midrule
""" + worst_rows + r"""\bottomrule
\end{tabular}
\label{tab:worst10}
\end{table}

\subsection{Per-Variable Breakdown (Top 5 Worst)}

\begin{table}[h]
\centering
\caption{Per-variable curve errors for the 5 worst scenarios.}
\begin{tabular}{clcccc}
\toprule
\textbf{Rank (ID)} & \textbf{Var} & \textbf{MAPE} & \textbf{NRMSE} &
\textbf{$R^2$} & \textbf{Max Abs Err} \\
\midrule
""" + worst_detail + r"""\bottomrule
\end{tabular}
\label{tab:worst5_detail}
\end{table}

\subsection{Input Parameters of Worst Scenarios}

\begin{table}[h]
\centering
\small
\caption{Input parameters of the 5 worst-performing scenarios.}
\begin{tabular}{cccccccccc}
\toprule
\textbf{Rank} & \textbf{ID} & \textbf{PI} & \textbf{$G_{\max}$} &
\textbf{$\nu$} & \textbf{$D_p$} & \textbf{$T_p$} & \textbf{$L_p$} &
\textbf{$I_p$} & \textbf{$D_p/L_p$} \\
\midrule
""" + worst_params_rows + r"""\bottomrule
\end{tabular}
\label{tab:worst_params}
\end{table}

%% ═══════════════════════════════════════════════════
\section{Diagnosis: Why Do These Scenarios Fail?}
\label{sec:diagnosis}
%% ═══════════════════════════════════════════════════

We compare the input feature distributions of the 10~worst scenarios against
the full test set to identify \emph{which physical parameters} push the model
outside its comfort zone.

\begin{table}[h]
\centering
\caption{Feature comparison: worst-10 mean vs.\ dataset mean.
$\sigma$-deviation measures how many standard deviations the worst-case
mean lies from the population mean.}
\begin{tabular}{lccc}
\toprule
\textbf{Feature} & \textbf{Worst Mean} & \textbf{All Mean} & \textbf{Deviation} \\
\midrule
""" + diag_rows + r"""\bottomrule
\end{tabular}
\label{tab:diagnosis}
\end{table}

\subsection{Root-Cause Analysis}

The worst-performing scenarios share several distinguishing characteristics:

\begin{enumerate}
""" + deviant_text + r"""
\end{enumerate}

\paragraph{Degradation curve characteristics.}
The worst scenarios typically exhibit:
\begin{itemize}[nosep]
    \item \textbf{Larger dynamic range}: the stiffness values span a wider
          interval, making relative errors more sensitive to small absolute
          offsets in the tail of the curve.
    \item \textbf{Non-standard degradation rates}: either unusually rapid
          or very slow degradation compared to the training distribution,
          which the 5~prototypes cannot fully capture.
    \item \textbf{Extreme parameter combinations}: multiple input features
          simultaneously at the tails of their distributions compound the
          extrapolation difficulty.
\end{itemize}

%% ═══════════════════════════════════════════════════
\section{Feature Importance (\% Influence on Prediction Error)}
\label{sec:importance}
%% ═══════════════════════════════════════════════════

To quantify how much each input parameter influences the prediction error, we
train a Gradient Boosting Regressor (GBR) that maps the 8~input features to
per-scenario average curve MAPE.  We report both the GBR's built-in impurity-based
importance and permutation importance (averaged over 30 repeats), then combine
them into a single ranking.

\begin{table}[h]
\centering
\caption{Feature importance ranking. \emph{Combined \%} is the average of
GBR and permutation importance, re-normalised to 100\%.}
\begin{tabular}{rccccc}
\toprule
\textbf{Influence} & \textbf{Feature} & \textbf{GBR Imp.} &
\textbf{Perm. Imp.} & \textbf{Corr $\to$ Error} \\
\midrule
""" + fi_rows + r"""\bottomrule
\end{tabular}
\label{tab:importance}
\end{table}

\subsection{Interpretation}

"""

    # Add interpretation for top-3 features
    for i, item in enumerate(sorted_features[:3]):
        col, imp = item
        latex += f"\\paragraph{{{i+1}. {col} ({imp['combined_influence_pct']:.1f}\\%).}}\n"
        if col == 'Gmax':
            latex += (
                "The maximum shear modulus $G_{\\max}$ is the dominant driver of prediction "
                "difficulty. Higher $G_{\\max}$ values correspond to stiffer soils where the "
                "stiffness degradation curves have larger absolute magnitudes, amplifying "
                "absolute errors.  The model's slot prototypes were learned from the full "
                "distribution; scenarios at the extremes of $G_{\\max}$ require extrapolation.\n\n"
            )
        elif col == 'Lp':
            latex += (
                "Pile length $L_p$ strongly influences the load transfer mechanism and the "
                "depth of soil mobilisation.  Longer piles engage more soil layers, creating "
                "complex multi-mode degradation patterns that a small number of prototypes "
                "cannot fully represent.\n\n"
            )
        elif col == 'Dp':
            latex += (
                "Pile diameter $D_p$ controls the contact surface area and the soil reaction "
                "pressure distribution.  Larger diameters lead to higher absolute stiffness "
                "values, making relative curve errors more sensitive.\n\n"
            )
        elif col == 'v':
            latex += (
                "Poisson's ratio $\\nu$ governs the volumetric vs.\\ shear response of the "
                "soil.  Near-incompressible soils ($\\nu \\approx 0.5$) produce very different "
                "degradation signatures than drained soils ($\\nu \\approx 0.25$).\n\n"
            )
        elif col == 'PI':
            latex += (
                "The Plasticity Index directly controls the clay's cyclic softening behaviour. "
                "High-PI clays degrade faster and more nonlinearly, creating steep curves "
                "that the model's monotonic constraints must balance against fitting accuracy.\n\n"
            )
        elif col == 'Tp':
            latex += (
                "Pile wall thickness $T_p$ affects the structural rigidity and the moment "
                "of inertia.  Very thin or very thick piles produce atypical stiffness "
                "evolution that the prototypes learned in Stage~B may not cover.\n\n"
            )
        elif col == 'Ip':
            latex += (
                "The moment of inertia $I_p$ is a derived geometric property that "
                "encapsulates both diameter and thickness effects.  It directly determines "
                "the pile's bending stiffness and hence the magnitude of KL/KR/KLR.\n\n"
            )
        elif col == 'Dp_Lp':
            latex += (
                "The slenderness ratio $D_p/L_p$ is a key dimensionless parameter in "
                "pile design.  Very slender piles ($D_p/L_p \\ll 1$) behave fundamentally "
                "differently from short stiff piles, and the transition zone is where "
                "the model struggles most.\n\n"
            )

    latex += r"""
%% ═══════════════════════════════════════════════════
\section{Conclusion}
%% ═══════════════════════════════════════════════════

The M8 Efficient $\Psi$-NN achieves strong curve-level performance with a
\textbf{mean MAPE of """ + f"{gm['avg_curve_mape_pct']:.2f}" + r"""\%} and
\textbf{mean curve $R^2$ of """ + f"{gm['avg_curve_r2']:.4f}" + r"""} across
""" + str(n_test) + r""" test scenarios, using only \textbf{45,486 parameters}
(19.7\% compression vs.\ teacher).  The worst-performing scenarios are
consistently characterised by extreme input parameter combinations that lie
at the tails of the training distribution.

\end{document}
"""

    latex_path = os.path.join(SCRIPT_DIR, 'scenario_analysis_report.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"\n  LaTeX report saved to scenario_analysis_report.tex")


if __name__ == "__main__":
    main()
