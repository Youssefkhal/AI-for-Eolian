"""
M9 Scenario Analysis: Curve-Based Errors, Worst-Case Diagnosis, Feature Importance
===================================================================================
- Loads the trained M9 SwiGLU Ψ-model (no retraining)
- Computes curve-level errors (cumulative stiffness curves, not per-slot drops)
- Identifies the worst scenarios and diagnoses why they fail
- Computes feature importance (% influence on prediction error)
- Exports results to JSON + generates a LaTeX report
"""

import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_STEPS = 21
STEP_INDICES = np.round(np.linspace(0, 43, NUM_STEPS)).astype(int)
SWIGLU_DIM = 32


class SwiGLUSlotMLP(nn.Module):
    def __init__(self, d_model=64, hidden=SWIGLU_DIM, dropout=0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, hidden)
        self.W_val = nn.Linear(d_model, hidden)
        self.W_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.W_out(self.drop(F.silu(self.W_gate(x)) * self.W_val(x)))


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
        self.slot_mlp = SwiGLUSlotMLP(d_model=d_model, hidden=SWIGLU_DIM, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.initial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))

    def get_relation_matrix(self):
        return torch.softmax(self.relation_logits, dim=1)

    def reconstruct_drop_slots(self, batch_size):
        protos = self.prototype_slots.expand(batch_size, -1, -1)
        relation = self.get_relation_matrix()
        drop_slots = torch.matmul(relation, protos)
        return drop_slots * self.slot_scales.unsqueeze(0)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = self.max_seq_len
        batch_size = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        initial = self.initial_slot.expand(batch_size, -1, -1)
        drops = self.reconstruct_drop_slots(batch_size)
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


def inverse_transform_values(scaled, scaler):
    flat = scaled.reshape(-1, 3)
    log_vals = scaler.inverse_transform(flat)
    orig = np.sign(log_vals) * np.expm1(np.abs(log_vals))
    return orig.reshape(scaled.shape)


def curve_mape(target_curve, pred_curve):
    denom = np.abs(target_curve)
    mask = denom > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(target_curve[mask] - pred_curve[mask]) / denom[mask]) * 100)


def curve_nrmse(target_curve, pred_curve):
    rmse = np.sqrt(np.mean((target_curve - pred_curve) ** 2))
    curve_range = target_curve.max() - target_curve.min()
    if curve_range < 1e-10:
        return 0.0
    return float(rmse / curve_range * 100)


def curve_r2(target_curve, pred_curve):
    if len(target_curve) < 2:
        return 1.0
    ss_res = np.sum((target_curve - pred_curve) ** 2)
    ss_tot = np.sum((target_curve - target_curve.mean()) ** 2)
    if ss_tot < 1e-20:
        return 1.0
    return float(1 - ss_res / ss_tot)


def curve_max_error(target_curve, pred_curve):
    return float(np.max(np.abs(target_curve - pred_curve)))


def area_between_curves(target_curve, pred_curve):
    diff = np.abs(target_curve - pred_curve)
    return float(np.trapezoid(diff, dx=1.0))


def generate_latex(results, input_cols, var_names, worst, diagnosis, sorted_features, n_test):
    gm = results['global_metrics']
    worst_rows = ""
    for rank, scenario in enumerate(worst, 1):
        overall = scenario['overall']
        worst_rows += (
            f"    {rank} & {scenario['scenario_id']} & {overall['avg_curve_mape_pct']:.2f}\\% & "
            f"{overall['avg_curve_nrmse_pct']:.2f}\\% & {overall['avg_curve_r2']:.4f} \\\\\n"
        )

    worst_detail = ""
    for rank, scenario in enumerate(worst[:5], 1):
        for index, var_name in enumerate(var_names):
            ve = scenario['per_variable'][var_name]
            prefix = f"\\multirow{{3}}{{*}}{{{rank} (S{scenario['scenario_id']})}}" if index == 0 else ""
            worst_detail += (
                f"    {prefix} & {var_name} & {ve['curve_mape_pct']:.2f}\\% & "
                f"{ve['curve_nrmse_pct']:.2f}\\% & {ve['curve_r2']:.4f} & "
                f"${ve['max_abs_error']:.2e}$ \\\\\n"
            )
        worst_detail += "    \\hline\n"

    fi_rows = ""
    for feature_name, importance in sorted_features:
        fi_rows += (
            f"    {importance['combined_influence_pct']:.1f}\\% & {feature_name} & "
            f"{importance['gbr_importance_pct']:.1f}\\% & "
            f"{importance['permutation_importance_pct']:.1f}\\% & "
            f"{importance['correlation_with_error']:+.4f} \\\\\n"
        )

    diag_rows = ""
    for col in input_cols:
        d = diagnosis[col]
        diag_rows += (
            f"    {col} & {d['worst_mean']:.4f} & {d['all_mean']:.4f} & "
            f"{d['deviation_sigma']:+.2f}$\\sigma$ \\\\\n"
        )

    pvd_rows = ""
    for var_name in var_names:
        pvd = results['per_variable_distribution'][var_name]
        pvd_rows += (
            f"    {var_name} & {pvd['mape_mean']:.2f}\\% & {pvd['mape_median']:.2f}\\% & "
            f"{pvd['mape_p95']:.2f}\\% & {pvd['r2_mean']:.4f} & {pvd['r2_min']:.4f} \\\\\n"
        )

    worst_params_rows = ""
    for rank, scenario in enumerate(worst[:5], 1):
        p = scenario['params']
        worst_params_rows += (
            f"    {rank} & S{scenario['scenario_id']} & {p['PI']:.0f} & {p['Gmax']:.0f} & "
            f"{p['v']:.3f} & {p['Dp']:.2f} & {p['Tp']:.4f} & {p['Lp']:.1f} & "
            f"{p['Ip']:.2e} & {p['Dp_Lp']:.4f} \\\\\n"
        )

    deviant_features = sorted(diagnosis.items(), key=lambda item: abs(item[1]['deviation_sigma']), reverse=True)
    deviant_text = ""
    for col, d in deviant_features[:3]:
        direction = "higher" if d['deviation_sigma'] > 0 else "lower"
        deviant_text += (
            f"\\item \\textbf{{{col}}}: The worst scenarios have a mean value of "
            f"{d['worst_mean']:.4f}, which is {abs(d['deviation_sigma']):.2f}$\\sigma$ "
            f"{direction} than the dataset mean ({d['all_mean']:.4f}).\n"
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

\title{M9 SwiGLU $\Psi$-NN: Curve-Based Error Analysis\\and Worst-Scenario Diagnosis}
\author{Automated Analysis Report}
\date{\today}

\begin{document}
\maketitle

\section{Model Overview}
The M9 SwiGLU $\Psi$-NN predicts lateral pile stiffness degradation curves over """ + str(gm['n_test']) + r""" test scenarios with
\textbf{45,502 parameters} (19.7\% fewer than the M6 teacher's 56,646).

\section{Global Curve-Based Performance}
\begin{table}[h]
\centering
\begin{tabular}{lccccc}
\toprule
\textbf{Variable} & \textbf{MAPE (mean)} & \textbf{MAPE (median)} & \textbf{MAPE (p95)} & \textbf{$R^2$ (mean)} & \textbf{$R^2$ (min)} \\
\midrule
""" + pvd_rows + r"""\bottomrule
\end{tabular}
\end{table}

Overall mean MAPE = """ + f"{gm['avg_curve_mape_pct']:.2f}" + r"""\%, mean $R^2$ = """ + f"{gm['avg_curve_r2']:.4f}" + r""".

\section{Worst-Performing Scenarios}
\begin{table}[h]
\centering
\begin{tabular}{ccccc}
\toprule
\textbf{Rank} & \textbf{Scenario} & \textbf{Avg MAPE} & \textbf{Avg NRMSE} & \textbf{Avg $R^2$} \\
\midrule
""" + worst_rows + r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Per-Variable Breakdown}
\begin{table}[h]
\centering
\begin{tabular}{clcccc}
\toprule
\textbf{Rank (ID)} & \textbf{Var} & \textbf{MAPE} & \textbf{NRMSE} & \textbf{$R^2$} & \textbf{Max Abs Err} \\
\midrule
""" + worst_detail + r"""\bottomrule
\end{tabular}
\end{table}

\subsection{Worst Scenario Inputs}
\begin{table}[h]
\centering
\small
\begin{tabular}{cccccccccc}
\toprule
\textbf{Rank} & \textbf{ID} & \textbf{PI} & \textbf{$G_{\max}$} & \textbf{$\nu$} & \textbf{$D_p$} & \textbf{$T_p$} & \textbf{$L_p$} & \textbf{$I_p$} & \textbf{$D_p/L_p$} \\
\midrule
""" + worst_params_rows + r"""\bottomrule
\end{tabular}
\end{table}

\section{Diagnosis}
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Feature} & \textbf{Worst Mean} & \textbf{All Mean} & \textbf{Deviation} \\
\midrule
""" + diag_rows + r"""\bottomrule
\end{tabular}
\end{table}

\begin{enumerate}
""" + deviant_text + r"""\end{enumerate}

\section{Feature Importance}
\begin{table}[h]
\centering
\begin{tabular}{rccccc}
\toprule
\textbf{Influence} & \textbf{Feature} & \textbf{GBR Imp.} & \textbf{Perm. Imp.} & \textbf{Corr $\to$ Error} \\
\midrule
""" + fi_rows + r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""

    latex_path = os.path.join(SCRIPT_DIR, 'scenario_analysis_report.tex')
    with open(latex_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write(latex)
    print("\n  LaTeX report saved to scenario_analysis_report.tex")


def main():
    print("=" * 70)
    print("M9 Scenario Analysis: Curve-Based Errors & Feature Importance")
    print("=" * 70)

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

    relation_matrix = np.array(psi_config['relation_matrix'])
    model = SlotAttentionPsiModel(
        input_size=len(feature_names), d_model=64, num_heads=4,
        max_seq_len=max_seq_len, dropout=0.1, num_iterations=3,
        num_prototypes=psi_config['num_prototypes'],
        relation_matrix=relation_matrix, structured=True)
    model.load_state_dict(torch.load(
        os.path.join(SCRIPT_DIR, 'pile_model.pth'), map_location='cpu', weights_only=True))
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_scaled), max_seq_len).numpy()
    pred_original = inverse_transform_values(pred_scaled, scaler_Y)

    n_test = len(X_original)
    print(f"Test scenarios: {n_test}")

    scenario_errors = []
    for index in range(n_test):
        target = Y_original[index]
        pred = pred_original[index]
        scenario = {
            'scenario_id': index + 1,
            'params': {col: float(X_original[index][j]) for j, col in enumerate(input_cols)},
        }

        var_errors = {}
        overall_mape = 0.0
        overall_nrmse = 0.0
        overall_r2 = 0.0
        overall_abc = 0.0
        for var_index, var_name in enumerate(var_names):
            target_curve = target[:, var_index]
            pred_curve = pred[:, var_index]
            mape = curve_mape(target_curve, pred_curve)
            nrmse = curve_nrmse(target_curve, pred_curve)
            r2 = curve_r2(target_curve, pred_curve)
            maxerr = curve_max_error(target_curve, pred_curve)
            abc = area_between_curves(target_curve, pred_curve)
            var_errors[var_name] = {
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

        scenario['per_variable'] = var_errors
        scenario['overall'] = {
            'avg_curve_mape_pct': round(overall_mape / 3, 3),
            'avg_curve_nrmse_pct': round(overall_nrmse / 3, 3),
            'avg_curve_r2': round(overall_r2 / 3, 6),
            'total_area_between_curves': float(f'{overall_abc:.4e}'),
        }
        scenario_errors.append(scenario)

    scenario_errors.sort(key=lambda item: item['overall']['avg_curve_mape_pct'], reverse=True)

    all_mapes = [scenario['overall']['avg_curve_mape_pct'] for scenario in scenario_errors]
    all_nrmses = [scenario['overall']['avg_curve_nrmse_pct'] for scenario in scenario_errors]
    all_r2s = [scenario['overall']['avg_curve_r2'] for scenario in scenario_errors]

    print(f"\n  Global Curve Metrics (across {n_test} test scenarios):")
    print(f"    Avg MAPE:  {np.mean(all_mapes):.2f}%  (median: {np.median(all_mapes):.2f}%)")
    print(f"    Avg NRMSE: {np.mean(all_nrmses):.2f}%  (median: {np.median(all_nrmses):.2f}%)")
    print(f"    Avg R²:    {np.mean(all_r2s):.4f}  (median: {np.median(all_r2s):.4f})")

    worst = scenario_errors[:10]
    best = scenario_errors[-20:]

    worst_params = np.array([[scenario['params'][col] for col in input_cols] for scenario in worst])
    best_params = np.array([[scenario['params'][col] for col in input_cols] for scenario in best])
    all_params = X_original.astype(float)

    diagnosis = {}
    for j, col in enumerate(input_cols):
        worst_mean = worst_params[:, j].mean()
        best_mean = best_params[:, j].mean()
        all_mean = all_params[:, j].mean()
        all_std = all_params[:, j].std()
        ratio = worst_mean / all_mean if abs(all_mean) > 1e-10 else 0.0
        deviation = (worst_mean - all_mean) / all_std if all_std > 1e-10 else 0.0
        diagnosis[col] = {
            'worst_mean': float(worst_mean),
            'best_mean': float(best_mean),
            'all_mean': float(all_mean),
            'all_std': float(all_std),
            'worst_to_all_ratio': float(ratio),
            'deviation_sigma': float(deviation),
        }

    error_by_id = {scenario['scenario_id']: scenario['overall']['avg_curve_mape_pct'] for scenario in scenario_errors}
    error_vec = np.array([error_by_id[i + 1] for i in range(n_test)])

    regressor = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, random_state=42, learning_rate=0.05, subsample=0.8)
    regressor.fit(X_original, error_vec)

    fi_builtin = regressor.feature_importances_
    fi_pct_builtin = fi_builtin / fi_builtin.sum() * 100
    perm_result = permutation_importance(regressor, X_original, error_vec, n_repeats=30, random_state=42)
    fi_perm = np.maximum(perm_result.importances_mean, 0)
    fi_pct_perm = fi_perm / fi_perm.sum() * 100 if fi_perm.sum() > 0 else fi_perm
    correlations = [float(np.corrcoef(X_original[:, j].astype(float), error_vec)[0, 1]) for j in range(len(input_cols))]

    combined = (fi_pct_builtin + fi_pct_perm) / 2
    combined_pct = combined / combined.sum() * 100
    importance_data = {}
    for j, col in enumerate(input_cols):
        importance_data[col] = {
            'gbr_importance_pct': round(float(fi_pct_builtin[j]), 2),
            'permutation_importance_pct': round(float(fi_pct_perm[j]), 2),
            'correlation_with_error': round(correlations[j], 4),
            'combined_influence_pct': round(float(combined_pct[j]), 2),
        }

    sorted_features = sorted(importance_data.items(), key=lambda item: item[1]['combined_influence_pct'], reverse=True)

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
        'best_scenarios': best[-5:],
        'diagnosis': diagnosis,
        'feature_importance': importance_data,
        'feature_ranking': [
            {'rank': i + 1, 'feature': col, 'influence_pct': imp['combined_influence_pct']}
            for i, (col, imp) in enumerate(sorted_features)
        ],
        'per_variable_distribution': {},
        'all_scenarios': scenario_errors,
    }

    for var_name in var_names:
        mapes = [scenario['per_variable'][var_name]['curve_mape_pct'] for scenario in scenario_errors]
        r2s = [scenario['per_variable'][var_name]['curve_r2'] for scenario in scenario_errors]
        results['per_variable_distribution'][var_name] = {
            'mape_mean': round(float(np.mean(mapes)), 3),
            'mape_median': round(float(np.median(mapes)), 3),
            'mape_p95': round(float(np.percentile(mapes, 95)), 3),
            'mape_max': round(float(np.max(mapes)), 3),
            'r2_mean': round(float(np.mean(r2s)), 6),
            'r2_median': round(float(np.median(r2s)), 6),
            'r2_p5': round(float(np.percentile(r2s, 5)), 6),
            'r2_min': round(float(np.min(r2s)), 6),
        }

    with open(os.path.join(SCRIPT_DIR, 'scenario_analysis.json'), 'w') as file_handle:
        json.dump(results, file_handle, indent=2)
    print("\n  Results saved to scenario_analysis.json")

    generate_latex(results, input_cols, var_names, worst, diagnosis, sorted_features, n_test)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
