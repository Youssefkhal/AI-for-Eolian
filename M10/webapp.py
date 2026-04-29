"""
M10: XAI-Enhanced SwiGLU Ψ-NN — Web Application
=================================================
Extends M9 with three XAI methods:
  - Token Attribution Maps (feature→slot importance)
  - Layer-wise Relevance Propagation (LRP to 8 raw input features)
  - Attention Rollout (cross + self separately)
  - Consistent curve-based error metrics throughout
"""
from flask import Flask, render_template_string, jsonify
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

NUM_STEPS = 21
SWIGLU_DIM = 32


import torch.nn.functional as F

from xai_engine import XAIAnalyzer

class SwiGLUSlotMLP(nn.Module):
    """SwiGLU slot MLP matching M9/train.py exactly."""
    def __init__(self, d_model=64, hidden=SWIGLU_DIM, dropout=0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, hidden)
        self.W_val  = nn.Linear(d_model, hidden)
        self.W_out  = nn.Linear(hidden,  d_model)
        self.drop   = nn.Dropout(dropout)
    def forward(self, x):
        return self.W_out(self.drop(F.silu(self.W_gate(x)) * self.W_val(x)))


class SlotAttentionPsiModel(nn.Module):
    """M9 SwiGLU Ψ-NN structured model (must match train.py exactly)."""

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


model, scaler_X, scaler_Y, feature_names, max_seq_len, test_data = None, None, None, None, None, None
psi_config, psi_discovery, comparison_data, adapter_comparison = None, None, None, None
scenarios_cache, metrics_cache = None, None
scenario_analysis = None


def _curve_mape(t, p):
    d = np.abs(t); mask = d > 1e-6
    return float(np.mean(np.abs(t[mask]-p[mask])/d[mask])*100) if mask.sum() else 0.0

def _curve_nrmse(t, p):
    rng = t.max()-t.min()
    return float(np.sqrt(np.mean((t-p)**2))/rng*100) if rng>1e-10 else 0.0

def _curve_r2(t, p):
    ss_res=np.sum((t-p)**2); ss_tot=np.sum((t-t.mean())**2)
    return float(1-ss_res/ss_tot) if ss_tot>1e-20 else 1.0


def inverse_transform(scaled, scaler_Y):
    flat = scaled.reshape(-1, 3)
    log_vals = scaler_Y.inverse_transform(flat)
    orig = np.sign(log_vals) * np.expm1(np.abs(log_vals))
    return orig.reshape(scaled.shape)


def calc_metrics(y_true, y_pred):
    ft, fp = y_true.flatten(), y_pred.flatten()
    if len(ft) < 2:
        return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}
    return {
        'r2': float(r2_score(ft, fp)),
        'rmse': float(np.sqrt(mean_squared_error(ft, fp))),
        'mae': float(mean_absolute_error(ft, fp)),
    }


def load_all():
    global model, scaler_X, scaler_Y, feature_names, max_seq_len, test_data
    global psi_config, psi_discovery, comparison_data, adapter_comparison
    global scenarios_cache, metrics_cache, scenario_analysis
    try:
        scaler_X = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
        scaler_Y = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
        feature_names = joblib.load(os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
        max_seq_len = joblib.load(os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
        test_data = joblib.load(os.path.join(SCRIPT_DIR, 'test_data.pkl'))
        psi_config = joblib.load(os.path.join(SCRIPT_DIR, 'psi_config.pkl'))

        discovery_path = os.path.join(SCRIPT_DIR, 'psi_discovery.json')
        if os.path.exists(discovery_path):
            with open(discovery_path) as f:
                psi_discovery = json.load(f)

        comparison_path = os.path.join(SCRIPT_DIR, 'comparison.json')
        if os.path.exists(comparison_path):
            with open(comparison_path) as f:
                comparison_data = json.load(f)

        adapter_path = os.path.join(SCRIPT_DIR, 'comparison_cross_attn_adapter.json')
        if os.path.exists(adapter_path):
            with open(adapter_path) as f:
                adapter_comparison = json.load(f)

        analysis_path = os.path.join(SCRIPT_DIR, 'scenario_analysis.json')
        if os.path.exists(analysis_path):
            import math
            def _clean_nan(obj):
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                if isinstance(obj, dict):
                    return {k: _clean_nan(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_clean_nan(v) for v in obj]
                return obj
            with open(analysis_path) as f:
                raw = f.read().replace('NaN', 'null').replace('Infinity', 'null')
                scenario_analysis = _clean_nan(json.loads(raw))

        R = np.array(psi_config['relation_matrix'])
        model = SlotAttentionPsiModel(
            input_size=len(feature_names), d_model=64, num_heads=4,
            max_seq_len=max_seq_len, dropout=0.1, num_iterations=3,
            num_prototypes=psi_config['num_prototypes'],
            relation_matrix=R, structured=True)
        model.load_state_dict(torch.load(
            os.path.join(SCRIPT_DIR, 'pile_model.pth'), map_location='cpu', weights_only=True))
        model.eval()

        X_scaled = test_data['X_scaled']
        X_original = test_data['X_original']
        Y_original = test_data['Y_original']
        input_cols = test_data['input_cols']

        with torch.no_grad():
            pred_scaled = model(torch.FloatTensor(X_scaled), max_seq_len).numpy()
        pred_original = inverse_transform(pred_scaled, scaler_Y)

        var_names = ['KL', 'KR', 'KLR']
        n_test = len(X_original)

        # Curve-based metrics — same aggregation as analyze_scenarios.py:
        # 1. Compute R²/MAPE/NRMSE per variable per scenario (full 21-step curve)
        # 2. Average the 3 variables per scenario → scenario-level metric
        # 3. Average across scenarios → global metric
        var_mapes = {vn: [] for vn in var_names}
        var_nrmses = {vn: [] for vn in var_names}
        var_r2s = {vn: [] for vn in var_names}
        scenario_avg_mapes = []
        scenario_avg_nrmses = []
        scenario_avg_r2s = []
        for i in range(n_test):
            sc_mape, sc_nrmse, sc_r2 = 0.0, 0.0, 0.0
            for vi, vn in enumerate(var_names):
                tc, pc = Y_original[i, :, vi], pred_original[i, :, vi]
                m = _curve_mape(tc, pc)
                n = _curve_nrmse(tc, pc)
                r = _curve_r2(tc, pc)
                var_mapes[vn].append(m)
                var_nrmses[vn].append(n)
                var_r2s[vn].append(r)
                sc_mape += m
                sc_nrmse += n
                sc_r2 += r
            scenario_avg_mapes.append(sc_mape / 3)
            scenario_avg_nrmses.append(sc_nrmse / 3)
            scenario_avg_r2s.append(sc_r2 / 3)

        m_overall = {
            'curve_mape': round(float(np.mean(scenario_avg_mapes)), 3),
            'curve_nrmse': round(float(np.mean(scenario_avg_nrmses)), 3),
            'curve_r2': round(float(np.mean(scenario_avg_r2s)), 6),
        }
        m_per_var = {}
        for vn in var_names:
            m_per_var[vn] = {
                'curve_mape': round(float(np.mean(var_mapes[vn])), 3),
                'curve_nrmse': round(float(np.mean(var_nrmses[vn])), 3),
                'curve_r2': round(float(np.mean(var_r2s[vn])), 6),
            }

        # Per-step metrics based on cumulative curve up to each step
        # (not individual slot values — uses the curve from step 1..s)
        m_per_slot = []
        for s in range(max_seq_len):
            step_metrics = {'slot': s + 1, 'type': 'initial' if s == 0 else 'drop'}
            step_errs = []
            step_var = {}
            for vi, vn in enumerate(var_names):
                # Use cumulative curve from step 0 to step s
                t_curve = Y_original[:, :s+1, vi]       # [n_test, s+1]
                p_curve = pred_original[:, :s+1, vi]     # [n_test, s+1]
                step_mapes = []
                for i in range(n_test):
                    step_mapes.append(_curve_mape(t_curve[i], p_curve[i]))
                step_mape = float(np.mean(step_mapes))
                step_var[vn] = {'mape': round(step_mape, 2)}
                step_errs.append(step_mape)
            step_metrics['avg_mape'] = round(float(np.mean(step_errs)), 2)
            step_metrics['per_variable'] = step_var
            m_per_slot.append(step_metrics)

        metrics_cache = {
            'overall': m_overall,
            'per_variable': m_per_var,
            'per_slot': m_per_slot,
        }

        scenarios_cache = []
        for i in range(n_test):
            params = {col: float(X_original[i, j]) for j, col in enumerate(input_cols)}
            y_t, y_p = Y_original[i], pred_original[i]
            # Curve-based metrics per variable
            curve_m = {}
            overall_mape, overall_nrmse, overall_r2 = 0.0, 0.0, 0.0
            for vi, vn in enumerate(var_names):
                tc, pc = y_t[:, vi], y_p[:, vi]
                cm = {'mape': round(_curve_mape(tc, pc), 2),
                      'nrmse': round(_curve_nrmse(tc, pc), 2),
                      'r2': round(_curve_r2(tc, pc), 4)}
                curve_m[vn] = cm
                overall_mape += cm['mape']
                overall_nrmse += cm['nrmse']
                overall_r2 += cm['r2']
            curve_m['avg_mape'] = round(overall_mape / 3, 2)
            curve_m['avg_nrmse'] = round(overall_nrmse / 3, 2)
            curve_m['avg_r2'] = round(overall_r2 / 3, 4)

            scenarios_cache.append({
                'id': i, 'params': params,
                'label': f"Scenario {i + 1}",
                'target': {vn: Y_original[i, :, vi].tolist() for vi, vn in enumerate(var_names)},
                'predicted': {vn: pred_original[i, :, vi].tolist() for vi, vn in enumerate(var_names)},
                'steps': list(range(1, max_seq_len + 1)),
                'curve_metrics': curve_m,
            })
        # Override comparison R² with curve-based values for consistency
        if comparison_data:
            for key in ['psi_model_m7', 'psi_model_m9']:
                if key in comparison_data and 'overall' in comparison_data[key]:
                    comparison_data[key]['overall']['r2'] = metrics_cache['overall']['curve_r2']
                    for vn in ['KL', 'KR', 'KLR']:
                        pv = comparison_data[key].get('per_variable', {})
                        if vn in pv:
                            pv[vn]['r2'] = metrics_cache['per_variable'][vn]['curve_r2']

        # ── Full sync: rebuild ALL scenario_analysis metrics from live predictions ──
        # This ensures the diagnosis page shows exactly the same numbers as the main dashboard.
        if scenario_analysis:
            # Build a lookup: scenario_id (1-indexed) → webapp curve_metrics
            sc_lookup = {}
            for sc in scenarios_cache:
                sid = sc['id'] + 1  # convert 0-indexed to 1-indexed
                sc_lookup[sid] = sc['curve_metrics']

            def _sync_scenario(sa_scenario):
                """Overwrite a scenario_analysis entry's metrics with webapp values."""
                sid = sa_scenario.get('scenario_id')
                if sid not in sc_lookup:
                    return
                cm = sc_lookup[sid]
                # Overwrite overall
                sa_scenario['overall'] = {
                    'avg_curve_mape_pct': cm['avg_mape'],
                    'avg_curve_nrmse_pct': cm['avg_nrmse'],
                    'avg_curve_r2': cm['avg_r2'],
                    'total_area_between_curves': sa_scenario.get('overall', {}).get('total_area_between_curves', 0),
                }
                # Overwrite per-variable
                for vn in ['KL', 'KR', 'KLR']:
                    if vn in cm and vn in sa_scenario.get('per_variable', {}):
                        sa_scenario['per_variable'][vn]['curve_mape_pct'] = cm[vn]['mape']
                        sa_scenario['per_variable'][vn]['curve_nrmse_pct'] = cm[vn]['nrmse']
                        sa_scenario['per_variable'][vn]['curve_r2'] = cm[vn]['r2']

            # Sync worst_scenarios, best_scenarios, all_scenarios
            for key in ['worst_scenarios', 'best_scenarios', 'all_scenarios']:
                for sa_sc in scenario_analysis.get(key, []):
                    _sync_scenario(sa_sc)

            # Re-sort worst by synced MAPE and take top 10
            if 'all_scenarios' in scenario_analysis:
                all_sc = scenario_analysis['all_scenarios']
                all_sc.sort(key=lambda s: s['overall']['avg_curve_mape_pct'], reverse=True)
                scenario_analysis['worst_scenarios'] = all_sc[:10]
                scenario_analysis['best_scenarios'] = all_sc[-5:]

            # Rebuild per_variable_distribution from synced values
            if 'all_scenarios' in scenario_analysis:
                for vn in ['KL', 'KR', 'KLR']:
                    mapes = []
                    r2s = []
                    for sa_sc in scenario_analysis['all_scenarios']:
                        pv = sa_sc.get('per_variable', {}).get(vn, {})
                        mapes.append(pv.get('curve_mape_pct', 0))
                        r2s.append(pv.get('curve_r2', 1))
                    scenario_analysis['per_variable_distribution'][vn] = {
                        'mape_mean': round(float(np.mean(mapes)), 3),
                        'mape_median': round(float(np.median(mapes)), 3),
                        'mape_p95': round(float(np.percentile(mapes, 95)), 3),
                        'mape_max': round(float(np.max(mapes)), 3),
                        'r2_mean': round(float(np.mean(r2s)), 6),
                        'r2_median': round(float(np.median(r2s)), 6),
                        'r2_p5': round(float(np.percentile(r2s, 5)), 6),
                        'r2_min': round(float(np.min(r2s)), 6),
                    }

            # Rebuild global_metrics
            if 'global_metrics' in scenario_analysis:
                gm = scenario_analysis['global_metrics']
                gm['avg_curve_r2'] = metrics_cache['overall']['curve_r2']
                gm['avg_curve_mape_pct'] = metrics_cache['overall']['curve_mape']
                gm['avg_curve_nrmse_pct'] = metrics_cache['overall']['curve_nrmse']
                # Also sync median and p95 from the synced scenario list
                if 'all_scenarios' in scenario_analysis:
                    all_mapes = [s['overall']['avg_curve_mape_pct']
                                 for s in scenario_analysis['all_scenarios']]
                    all_r2s = [s['overall']['avg_curve_r2']
                               for s in scenario_analysis['all_scenarios']]
                    gm['median_curve_mape_pct'] = round(float(np.median(all_mapes)), 3)
                    gm['median_curve_r2'] = round(float(np.median(all_r2s)), 6)
                    gm['p95_curve_mape_pct'] = round(float(np.percentile(all_mapes, 95)), 3)

        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return False


HTML = '''<!DOCTYPE html><html><head><title>M10 XAI-Enhanced Ψ-NN Pile Stiffness</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);min-height:100vh;color:#fff;padding:20px}
.container{max-width:1600px;margin:0 auto}
h1{text-align:center;font-size:1.5rem;margin-bottom:6px;color:#00d2ff}
.subtitle{text-align:center;font-size:0.78rem;color:#88c8e8;margin-bottom:16px}
h2{color:#00d2ff;margin-bottom:10px;font-size:0.95rem}
h3{color:#00d2ff;margin:14px 0 8px;font-size:0.88rem}
.card{background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;border:1px solid rgba(255,255,255,0.1);margin-bottom:12px}

.arch-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.arch-box{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:10px}
.arch-title{font-size:0.82rem;color:#ffa500;font-weight:700;margin-bottom:5px}
.arch-line{font-size:0.74rem;color:#ddd;padding:1px 0}
.arch-line .k{color:#88c8e8}
.arch-chip{display:inline-block;background:rgba(0,210,255,0.1);border:1px solid rgba(0,210,255,0.2);border-radius:5px;padding:3px 8px;font-size:0.72rem;margin-right:6px;margin-bottom:6px}
.arch-note{font-size:0.72rem;color:#aaa;line-height:1.45}

.psi-banner{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.psi-stat{background:rgba(255,165,0,0.08);border:1px solid rgba(255,165,0,0.25);border-radius:10px;padding:12px 16px;text-align:center}
.psi-stat .label{font-size:0.7rem;color:#f0a050;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.psi-stat .value{font-size:1.3rem;font-weight:700;color:#ffa500}
.psi-stat .sub{font-size:0.68rem;color:#886;margin-top:2px}

.comp-table{width:100%;border-collapse:collapse;font-size:0.76rem;margin-bottom:12px}
.comp-table th{background:rgba(255,165,0,0.1);color:#ffa500;padding:8px 12px;text-align:center}
.comp-table td{padding:6px 12px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:center}
.comp-table tr:hover{background:rgba(255,165,0,0.05)}
.comp-best{color:#00ff88;font-weight:700}

.cluster-row{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.cluster-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:10px 14px;flex:1;min-width:180px}
.cluster-card .cn{font-weight:700;color:#ffa500;margin-bottom:4px;font-size:0.85rem}
.cluster-card .cm{font-size:0.72rem;color:#aaa}
.cluster-card .cs{font-size:0.72rem;color:#00ff88;margin-top:3px}

.heatmap-wrap{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.heatmap-box{flex:1;min-width:300px}
.heatmap{display:grid;font-size:0.6rem;gap:1px;background:rgba(0,0,0,0.3);border-radius:4px;overflow:hidden}
.heatmap-cell{padding:3px;text-align:center;min-width:28px}
.heatmap-header{background:rgba(0,210,255,0.15);color:#00d2ff;font-weight:700;font-size:0.6rem}

.metrics-banner{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.metric-big{flex:1;min-width:200px;background:rgba(0,210,255,0.08);border:1px solid rgba(0,210,255,0.25);border-radius:10px;padding:14px 18px;text-align:center}
.metric-big .label{font-size:0.72rem;color:#88c8e8;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.metric-big .value{font-size:1.5rem;font-weight:700;color:#00d2ff}
.metric-big .sub{font-size:0.7rem;color:#668;margin-top:2px}

.var-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
.var-card{background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;border:1px solid rgba(255,255,255,0.08)}
.var-card .vname{font-weight:700;color:#00ff88;margin-bottom:6px;font-size:0.9rem}
.var-card .vrow{display:flex;justify-content:space-between;font-size:0.75rem;padding:2px 0}
.var-card .vrow .vl{color:#888}.var-card .vrow .vv{color:#fff;font-weight:600}

.grid{display:grid;grid-template-columns:260px 1fr;gap:12px}
.sc-list{list-style:none;max-height:55vh;overflow-y:auto}
.sc-list li{padding:8px 10px;border-radius:6px;cursor:pointer;margin-bottom:3px;font-size:0.78rem;border:1px solid transparent;transition:all .2s}
.sc-list li:hover{background:rgba(0,210,255,0.1);border-color:rgba(0,210,255,0.3)}
.sc-list li.active{background:rgba(0,210,255,0.2);border-color:#00d2ff;color:#00d2ff;font-weight:600}
.sc-params{font-size:0.65rem;color:#888;margin-top:2px}
.sc-r2{font-size:0.65rem;color:#00ff88;margin-top:1px}

.params-bar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px}
.param-chip{background:rgba(0,210,255,0.1);border:1px solid rgba(0,210,255,0.2);border-radius:4px;padding:3px 7px;font-size:0.72rem}
.param-chip .lbl{color:#888}.param-chip .val{color:#00d2ff;font-weight:600}

.sc-metrics-bar{display:flex;gap:16px;margin-bottom:10px;padding:8px 12px;background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.15);border-radius:6px;flex-wrap:wrap}
.sc-metric{font-size:0.78rem}.sc-metric .ml{color:#888}.sc-metric .mv{color:#00ff88;font-weight:600}

.sc-var-row{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}
.sc-var-chip{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:6px 10px;font-size:0.72rem;flex:1;min-width:180px}
.sc-var-chip strong{color:#00d2ff}

.charts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px}
.chc{background:rgba(255,255,255,0.03);border-radius:8px;padding:10px}
.cht{color:#00d2ff;margin-bottom:6px;font-size:0.82rem;font-weight:600;text-align:center}
canvas{width:100%!important;height:250px!important}

.legend-bar{display:flex;justify-content:center;gap:20px;margin-bottom:10px}
.legend-item{display:flex;align-items:center;gap:5px;font-size:0.75rem}
.legend-dot{width:12px;height:3px;border-radius:2px}

.table-wrap{max-height:400px;overflow:auto;margin-bottom:12px}
table.step-tbl{width:100%;border-collapse:collapse;font-size:0.72rem}
table.step-tbl th{background:rgba(0,210,255,0.1);color:#00d2ff;position:sticky;top:0;padding:6px 8px;text-align:right;white-space:nowrap}
table.step-tbl th:first-child,table.step-tbl th:nth-child(2){text-align:center}
table.step-tbl td{padding:5px 8px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:right;white-space:nowrap}
table.step-tbl td:first-child,table.step-tbl td:nth-child(2){text-align:center}
table.step-tbl tr:hover{background:rgba(0,210,255,0.05)}
.err-good{color:#00ff88}.err-mid{color:#ffa500}.err-bad{color:#ff5757}

.slot-section{margin-top:16px}
table.slot-tbl{width:100%;border-collapse:collapse;font-size:0.73rem}
table.slot-tbl th{background:rgba(0,210,255,0.1);color:#00d2ff;padding:6px 10px;text-align:center;position:sticky;top:0}
table.slot-tbl td{padding:5px 10px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:center}
table.slot-tbl tr:hover{background:rgba(0,210,255,0.05)}

.st{padding:6px 12px;border-radius:4px;margin-bottom:12px;text-align:center;font-size:0.8rem}
.st.ok{background:rgba(0,255,136,0.1);color:#00ff88}
.st.err{background:rgba(255,87,87,0.1);color:#ff5757}
.empty{text-align:center;color:#666;padding:40px;font-size:0.9rem}
@media(max-width:1100px){.grid{grid-template-columns:1fr}.charts{grid-template-columns:1fr}.var-metrics{grid-template-columns:1fr}.psi-banner{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<div class="container">
<h1>M10: SwiGLU &Psi;-NN Pile Stiffness Degradation</h1>
<p class="subtitle">SwiGLU Slot MLP, Learnable R, Physics Loss &nbsp;|&nbsp; <a href="/xai" style="color:#c084fc;text-decoration:underline">XAI Dashboard &rarr;</a> &nbsp;|&nbsp; <a href="/diagnosis" style="color:#ffa500;text-decoration:underline">Worst-Scenario Diagnosis &rarr;</a></p>
<div id="st" class="st ok">Loading model...</div>

<div id="psiSection"></div>
<div id="archSection"></div>
<div class="metrics-banner" id="overallMetrics"></div>
<div class="var-metrics" id="varMetrics"></div>

<div class="grid">
<div class="card">
<h2>Test Scenarios</h2>
<ul class="sc-list" id="scList"></ul>
</div>
<div class="card">
<div id="detail"><div class="empty">Select a scenario from the list</div></div>
</div>
</div>

<div class="card slot-section" id="slotSection">
<h2>Per-Step Cumulative Curve MAPE (Across All Test Scenarios)</h2>
<div class="table-wrap" id="slotTable"></div>
</div>
</div>

<script>
let scenarios=[], metrics={}, psiInfo={}, charts={};
const fmt=n=>{const a=Math.abs(n);if(a>=1e9)return(n/1e9).toFixed(2)+'e9';if(a>=1e6)return(n/1e6).toFixed(2)+'e6';if(a>=1e3)return(n/1e3).toFixed(2)+'e3';if(a<0.01&&a>0)return n.toExponential(2);return n.toFixed(2)};
const fmtM=n=>n.toFixed(4);
const fmtS=n=>n.toExponential(3);
const errPct=(t,p)=>{if(Math.abs(t)<1e-10)return'-';return(Math.abs(p-t)/Math.abs(t)*100).toFixed(1)};
const errClass=e=>{if(e==='-')return'';const v=parseFloat(e);if(v<5)return'err-good';if(v<20)return'err-mid';return'err-bad'};

function mkChart(id,label,tgt,pred,steps){
    if(charts[id])charts[id].destroy();
    const ctx=document.getElementById(id);
    charts[id]=new Chart(ctx,{type:'line',data:{labels:steps,datasets:[
        {label:'Target',data:tgt,borderColor:'#00ff88',fill:false,tension:.3,pointRadius:2,pointBackgroundColor:'#00ff88',borderWidth:2},
        {label:'Predicted',data:pred,borderColor:'#ff6b6b',fill:false,tension:.3,pointRadius:2,pointBackgroundColor:'#ff6b6b',borderWidth:2,borderDash:[5,3]}
    ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,callbacks:{label:c=>c.dataset.label+': '+fmt(c.parsed.y)}}},scales:{x:{title:{display:true,text:'Step',color:'#aaa'},grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#aaa'}},y:{title:{display:true,text:label,color:'#aaa'},grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#aaa',callback:v=>fmt(v)}}},interaction:{mode:'nearest',axis:'x',intersect:false}}});
}

function renderPsiSection(psi){
    const el=document.getElementById('psiSection');
    if(!psi||!psi.discovery){el.innerHTML='';return;}
    const d=psi.discovery, comp=psi.comparison;
    let h='<div class="card"><h2>&Psi;-NN Structure Discovery</h2>';
    h+='<div class="psi-banner">';
    h+=`<div class="psi-stat"><div class="label">Prototypes (k*)</div><div class="value">${d.k_star}</div><div class="sub">from 20 drop slots</div></div>`;
    h+=`<div class="psi-stat"><div class="label">Silhouette Score</div><div class="value">${d.best_silhouette.toFixed(3)}</div><div class="sub">clustering quality</div></div>`;
    if(comp&&comp.psi_model_m7){
        h+=`<div class="psi-stat"><div class="label">Compression</div><div class="value">${comp.psi_model_m7.compression}</div><div class="sub">fewer params vs M6</div></div>`;
        h+=`<div class="psi-stat"><div class="label">&Psi;-Model Params</div><div class="value">${comp.psi_model_m7.params.toLocaleString()}</div><div class="sub">vs ${comp.teacher_m6.params.toLocaleString()} (M6)</div></div>`;
    }
    h+='</div>';
    if(comp){
        h+='<h3>Model Comparison: M6 Teacher vs &Psi;-Model (M7)</h3>';
        h+='<table class="comp-table"><thead><tr><th>Model</th><th>Params</th><th>R&sup2;</th>';
        for(const vn of ['KL','KR','KLR'])h+=`<th>${vn} R&sup2;</th>`;
        h+='</tr></thead><tbody>';
        const models=[['M6 Teacher',comp.teacher_m6],['Stage-A Student',comp.student_stage_a],['&Psi;-Model (M7)',comp.psi_model_m7]];
        const bestR2=Math.max(...models.map(m=>m[1].overall.r2));
        for(const [name,m] of models){
            const isBest=m.overall.r2===bestR2;
            h+=`<tr><td><strong>${name}</strong></td><td>${m.params.toLocaleString()}</td>`;
            h+=`<td class="${isBest?'comp-best':''}">${fmtM(m.overall.r2)}</td>`;
            for(const vn of ['KL','KR','KLR'])h+=`<td>${fmtM(m.per_variable[vn].r2)}</td>`;
            h+='</tr>';
        }
        h+='</tbody></table>';
    }
    if(d.cluster_info){
        h+='<h3>Discovered Prototype Clusters</h3><div class="cluster-row">';
        const sortedP=Object.entries(d.cluster_info).sort((a,b)=>a[0].localeCompare(b[0]));
        const colors=['#ff6b6b','#4dc9f6','#ffa500','#00ff88','#c084fc','#f472b6','#a3e635','#38bdf8','#fb923c','#e879f9'];
        sortedP.forEach(([name,info],pi)=>{
            const c=colors[pi%colors.length];
            h+=`<div class="cluster-card" style="border-color:${c}40">`;
            h+=`<div class="cn" style="color:${c}">${name.replace('_',' ')}</div>`;
            h+=`<div class="cm">Slots: ${info.members.join(', ')}</div>`;
            h+=`<div class="cs">${info.count} members &middot; norm: ${info.avg_norm.toFixed(3)}</div>`;
            h+='</div>';
        });
        h+='</div>';
    }
    if(d.silhouettes){
        h+='<h3>Silhouette Analysis</h3>';
        h+='<div style="max-width:500px;background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;margin-bottom:12px">';
        h+='<canvas id="cSilhouette" style="height:200px!important"></canvas></div>';
    }
    if(d.cosine_similarity){
        h+='<h3>Drop-Slot Cosine Similarity</h3>';
        const sim=d.cosine_similarity;const n=sim.length;const cols=n+1;
        h+=`<div class="heatmap-wrap"><div class="heatmap-box"><div class="heatmap" style="grid-template-columns:repeat(${cols},1fr)">`;
        h+='<div class="heatmap-cell heatmap-header"></div>';
        for(let j=0;j<n;j++)h+=`<div class="heatmap-cell heatmap-header">S${j+2}</div>`;
        for(let i=0;i<n;i++){
            h+=`<div class="heatmap-cell heatmap-header">S${i+2}</div>`;
            for(let j=0;j<n;j++){
                const v=sim[i][j];
                const r=Math.round(Math.max(0,v)*255);
                const b=Math.round(Math.max(0,-v)*255);
                const bg=v>=0?`rgba(${r},${Math.round(r*0.6)},0,${Math.abs(v)*0.8})`:`rgba(0,${Math.round(b*0.4)},${b},${Math.abs(v)*0.8})`;
                h+=`<div class="heatmap-cell" style="background:${bg}" title="S${i+2}&times;S${j+2}=${v.toFixed(2)}">${v.toFixed(1)}</div>`;
            }
        }
        h+='</div></div></div>';
    }
    h+='</div>';
    el.innerHTML=h;
    if(d.silhouettes){
        const ks=Object.keys(d.silhouettes).map(Number);
        const sils=ks.map(k=>d.silhouettes[String(k)]);
        const ctx=document.getElementById('cSilhouette');
        new Chart(ctx,{type:'bar',data:{labels:ks.map(k=>'k='+k),datasets:[
            {label:'Silhouette',data:sils,backgroundColor:ks.map(k=>k===d.k_star?'rgba(255,165,0,0.8)':'rgba(255,165,0,0.3)'),borderColor:'#ffa500',borderWidth:1}
        ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#aaa'}},x:{grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#aaa'}}}}});
    }
}

function renderArchitectureSection(psi){
        const el=document.getElementById('archSection');
        const comp=psi&&psi.comparison?psi.comparison:null;
        const adap=psi&&psi.adapter_comparison?psi.adapter_comparison:null;
        let h='<div class="card"><h2>M9 Detailed Architecture Schematic (with Added Cross-Attention)</h2>';
        h+='<div class="arch-grid">';
        h+='<div class="arch-box">';
        h+='<div class="arch-title">Base M9 Pipeline (kept intact)</div>';
        h+='<div class="arch-line"><span class="k">Input:</span> 8 features</div>';
        h+='<div class="arch-line"><span class="k">Embedding:</span> Linear(8-&gt;64) + LayerNorm + GELU</div>';
        h+='<div class="arch-line"><span class="k">Slots:</span> 1 initial + discovered drop slots</div>';
        h+='<div class="arch-line"><span class="k">Core:</span> Cross-Attn + Self-Attn + SwiGLU MLP</div>';
        h+='<div class="arch-line"><span class="k">Relation Matrix:</span> Learnable logits -&gt; softmax</div>';
        h+='<div class="arch-line"><span class="k">Output:</span> KL, KR, KLR across 21 steps</div>';
        h+='</div>';
        h+='<div class="arch-box">';
        h+='<div class="arch-title">Added Module (new trainable part only)</div>';
        h+='<div class="arch-line"><span class="k">Adapter:</span> Decoder-style cross-attention</div>';
        h+='<div class="arch-line"><span class="k">Query:</span> Learned decoder queries</div>';
        h+='<div class="arch-line"><span class="k">Key/Value:</span> Frozen M9 slot representations</div>';
        h+='<div class="arch-line"><span class="k">Fusion:</span> Gated residual + LayerNorm</div>';
        h+='<div class="arch-line"><span class="k">Train Scope:</span> Adapter only (base M9 frozen)</div>';
        h+='</div>';
        h+='</div>';
        h+='<div style="margin-bottom:8px">';
        h+='<span class="arch-chip">SwiGLU slot MLP</span>';
        h+='<span class="arch-chip">Learnable relation matrix (logits-&gt;softmax)</span>';
        h+='<span class="arch-chip">Physics-consistent trend constraints</span>';
        h+='</div>';
        if(comp&&comp.psi_model_m9&&comp.psi_model_m9.overall){
                const b=comp.psi_model_m9.overall;
                h+='<div class="arch-note" style="margin-bottom:6px">Base M9: '
                    +`R&sup2;=${b.r2.toFixed(4)} | RMSE=${fmt(b.rmse)} | MAE=${fmt(b.mae)}</div>`;
        }
        if(adap&&adap.base_m9&&adap.base_m9.overall&&adap.m9_cross_attn_adapter&&adap.m9_cross_attn_adapter.overall){
                const b=adap.base_m9.overall;
                const a=adap.m9_cross_attn_adapter.overall;
                h+='<div class="arch-note">Adapter experiment: '
                    +`Base R&sup2;=${b.r2.toFixed(4)} -&gt; Adapter R&sup2;=${a.r2.toFixed(4)} | `
                    +`Trainable params=${adap.m9_cross_attn_adapter.params_trainable.toLocaleString()}</div>`;
        }
        h+='<div class="arch-note" style="margin-top:8px">Overleaf file: <strong>m9_architecture_schematic_overleaf.tex</strong></div>';
        h+='</div>';
        el.innerHTML=h;
}

function showScenario(idx){
    document.querySelectorAll('.sc-list li').forEach((li,i)=>li.classList.toggle('active',i===idx));
    const s=scenarios[idx];
    const pn={'PI':'PI','Gmax':'Gmax (Pa)','v':'v','Dp':'Dp (m)','Tp':'Tp (m)','Lp':'Lp (m)','Ip':'Ip','Dp_Lp':'Dp/Lp'};
    let h='<div class="params-bar">';
    for(const[k,v]of Object.entries(s.params))h+=`<div class="param-chip"><span class="lbl">${pn[k]||k}:</span> <span class="val">${fmt(v)}</span></div>`;
    h+='</div>';
    h+='<div class="sc-metrics-bar">';
    h+=`<div class="sc-metric"><span class="ml">Curve MAPE = </span><span class="mv" style="color:${s.curve_metrics.avg_mape<5?'#00ff88':s.curve_metrics.avg_mape<15?'#ffa500':'#ff5757'}">${s.curve_metrics.avg_mape}%</span></div>`;
    h+=`<div class="sc-metric"><span class="ml">Curve NRMSE = </span><span class="mv">${s.curve_metrics.avg_nrmse}%</span></div>`;
    h+=`<div class="sc-metric"><span class="ml">Curve R&sup2; = </span><span class="mv">${s.curve_metrics.avg_r2.toFixed(4)}</span></div>`;
    h+='</div>';
    h+='<div class="sc-var-row">';
    for(const vn of ['KL','KR','KLR']){const cm=s.curve_metrics[vn];h+=`<div class="sc-var-chip"><strong>${vn}</strong> &nbsp; MAPE=${cm.mape}% &nbsp; NRMSE=${cm.nrmse}% &nbsp; R&sup2;=${cm.r2.toFixed(4)}</div>`;}
    h+='</div>';
    h+='<div class="legend-bar"><div class="legend-item"><div class="legend-dot" style="background:#00ff88"></div>Target</div><div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div>Predicted (&Psi;-Model)</div></div>';
    h+='<div class="charts">';
    h+='<div class="chc"><div class="cht">KL</div><canvas id="cKL"></canvas></div>';
    h+='<div class="chc"><div class="cht">KR</div><canvas id="cKR"></canvas></div>';
    h+='<div class="chc"><div class="cht">KLR</div><canvas id="cKLR"></canvas></div>';
    h+='</div>';
    h+='<h3>Per-Step Values</h3><div class="table-wrap"><table class="step-tbl"><thead><tr><th>Step</th><th>Type</th>';
    for(const vn of ['KL','KR','KLR'])h+=`<th>${vn} Target</th><th>${vn} Pred</th><th>${vn} Err%</th>`;
    h+='</tr></thead><tbody>';
    for(let st=0;st<s.steps.length;st++){
        h+='<tr><td>'+s.steps[st]+'</td><td>'+(st===0?'Initial':'Drop '+st)+'</td>';
        for(const vn of ['KL','KR','KLR']){const t=s.target[vn][st],p=s.predicted[vn][st],e=errPct(t,p);h+=`<td>${fmt(t)}</td><td>${fmt(p)}</td><td class="${errClass(e)}">${e}%</td>`;}
        h+='</tr>';
    }
    h+='</tbody></table></div>';
    document.getElementById('detail').innerHTML=h;
    mkChart('cKL','KL',s.target.KL,s.predicted.KL,s.steps);
    mkChart('cKR','KR',s.target.KR,s.predicted.KR,s.steps);
    mkChart('cKLR','KLR',s.target.KLR,s.predicted.KLR,s.steps);
}

function renderOverall(metrics){
    const m=metrics.overall;const pv=metrics.per_variable;
    const cm=(m.curve_r2!==undefined)?{mape:m.curve_mape.toFixed(2),nrmse:m.curve_nrmse.toFixed(2),r2:m.curve_r2.toFixed(4)}:{mape:'--',nrmse:'--',r2:'--'};
    document.getElementById('overallMetrics').innerHTML=`
        <div class="metric-big" style="background:rgba(255,165,0,0.08);border-color:rgba(255,165,0,0.25)"><div class="label" style="color:#f0a050">Curve MAPE</div><div class="value" style="color:#ffa500">${cm.mape}%</div><div class="sub">Mean Abs % Error (target vs predicted curves)</div></div>
        <div class="metric-big" style="background:rgba(255,165,0,0.08);border-color:rgba(255,165,0,0.25)"><div class="label" style="color:#f0a050">Curve NRMSE</div><div class="value" style="color:#ffa500">${cm.nrmse}%</div><div class="sub">Normalized RMSE (curve range)</div></div>
        <div class="metric-big" style="background:rgba(255,165,0,0.08);border-color:rgba(255,165,0,0.25)"><div class="label" style="color:#f0a050">Curve R&sup2;</div><div class="value" style="color:#ffa500">${cm.r2}</div><div class="sub">Avg per-variable curve fit</div></div>`;
}

function renderVarMetrics(mv){
    let h='';const colors={'KL':'#00ff88','KR':'#4dc9f6','KLR':'#f67019'};
    for(const vn of ['KL','KR','KLR']){const m=mv[vn];h+=`<div class="var-card"><div class="vname" style="color:${colors[vn]}">${vn}</div><div class="vrow"><span class="vl">MAPE</span><span class="vv">${m.curve_mape}%</span></div><div class="vrow"><span class="vl">NRMSE</span><span class="vv">${m.curve_nrmse}%</span></div><div class="vrow"><span class="vl">R&sup2;</span><span class="vv">${m.curve_r2.toFixed(4)}</span></div></div>`;}
    document.getElementById('varMetrics').innerHTML=h;
}

function renderSlotMetrics(slots){
    let h='<table class="slot-tbl"><thead><tr><th>Step</th><th>Type</th><th>Avg MAPE</th>';
    for(const vn of ['KL','KR','KLR'])h+=`<th>${vn} MAPE</th>`;
    h+='</tr></thead><tbody>';
    for(const s of slots){
        h+=`<tr><td>${s.slot}</td><td>${s.type==='initial'?'Initial':'Drop '+(s.slot-1)}</td>`;
        h+=`<td class="${s.avg_mape<2?'err-good':s.avg_mape<5?'':'err-mid'}">${s.avg_mape}%</td>`;
        for(const vn of ['KL','KR','KLR']){h+=`<td>${s.per_variable[vn].mape}%</td>`;}
        h+='</tr>';
    }
    h+='</tbody></table>';
    document.getElementById('slotTable').innerHTML=h;
}

window.onload=async()=>{
    const st=document.getElementById('st');
    try{
        const r=await(await fetch('/api/scenarios')).json();
        if(r.error){st.textContent=r.error;st.className='st err';return}
        scenarios=r.scenarios;metrics=r.metrics;psiInfo=r.psi||{};
        const kStar=psiInfo.discovery?psiInfo.discovery.k_star:'?';
        st.textContent='\\u03A8-Model loaded \\u2014 k*='+kStar+' prototypes \\u2014 '+scenarios.length+' test scenarios \\u2014 '+r.num_steps+' steps';
        renderPsiSection(psiInfo);
        renderArchitectureSection(psiInfo);
        renderOverall(metrics);
        renderVarMetrics(metrics.per_variable);
        renderSlotMetrics(metrics.per_slot);
        const list=document.getElementById('scList');
        scenarios.forEach((s,i)=>{
            const li=document.createElement('li');
            li.innerHTML=`<div>${s.label}</div><div class="sc-params">PI=${fmt(s.params.PI)} Gmax=${fmt(s.params.Gmax)} Lp=${fmt(s.params.Lp)}</div><div class="sc-r2" style="color:${s.curve_metrics.avg_mape<5?'#00ff88':s.curve_metrics.avg_mape<15?'#ffa500':'#ff5757'}">MAPE=${s.curve_metrics.avg_mape}% &nbsp; R&sup2;=${s.curve_metrics.avg_r2.toFixed(4)}</div>`;
            li.onclick=()=>showScenario(i);
            list.appendChild(li);
        });
        // Check URL hash for scenario redirect (e.g. /#scenario=56)
        const hash=window.location.hash;
        const hm=hash.match(/scenario=(\d+)/);
        if(hm){
            const sid=parseInt(hm[1]);
            const idx=scenarios.findIndex(s=>s.id===sid);
            if(idx>=0){showScenario(idx);document.querySelectorAll('.sc-list li')[idx].scrollIntoView({block:'center'});}
            else if(scenarios.length>0)showScenario(0);
        } else if(scenarios.length>0)showScenario(0);
    }catch(e){st.textContent='Failed: '+e;st.className='st err'}
};
</script></body></html>'''


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/diagnosis')
def diagnosis_page():
    return render_template_string(DIAGNOSIS_HTML)


@app.route('/api/scenarios')
def api_scenarios():
    if scenarios_cache is None and not load_all():
        return jsonify({'error': 'Model not loaded. Run train.py first.'})
    psi = {}
    if psi_discovery:
        psi['discovery'] = psi_discovery
    if comparison_data:
        psi['comparison'] = comparison_data
    if adapter_comparison:
        psi['adapter_comparison'] = adapter_comparison
    return jsonify({
        'scenarios': scenarios_cache,
        'metrics': metrics_cache,
        'num_steps': max_seq_len,
        'psi': psi,
    })


# ─────────────────────────────────────────────────────
# XAI Dashboard
# ─────────────────────────────────────────────────────

XAI_HTML = r'''<!DOCTYPE html><html><head><title>M10 XAI Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);min-height:100vh;color:#fff;padding:20px}
.container{max-width:1600px;margin:0 auto}
h1{text-align:center;font-size:1.5rem;margin-bottom:6px;color:#c084fc}
.subtitle{text-align:center;font-size:0.78rem;color:#d8b4fe;margin-bottom:16px}
h2{color:#c084fc;margin-bottom:10px;font-size:0.95rem}
h3{color:#c084fc;margin:14px 0 8px;font-size:0.88rem}
.card{background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;border:1px solid rgba(255,255,255,0.1);margin-bottom:12px}
.back-link{display:inline-block;margin-bottom:14px;color:#c084fc;text-decoration:underline;font-size:0.85rem}
.back-link:hover{color:#fff}
.st{padding:6px 12px;border-radius:4px;margin-bottom:12px;text-align:center;font-size:0.8rem}
.st.ok{background:rgba(192,132,252,0.1);color:#c084fc}

.grid{display:grid;grid-template-columns:260px 1fr;gap:12px}
.sc-list{list-style:none;max-height:70vh;overflow-y:auto}
.sc-list li{padding:8px 10px;border-radius:6px;cursor:pointer;margin-bottom:3px;font-size:0.78rem;border:1px solid transparent;transition:all .2s}
.sc-list li:hover{background:rgba(192,132,252,0.1);border-color:rgba(192,132,252,0.3)}
.sc-list li.active{background:rgba(192,132,252,0.2);border-color:#c084fc;color:#c084fc;font-weight:600}
.sc-params{font-size:0.65rem;color:#888;margin-top:2px}

.var-tabs{display:flex;gap:6px;margin-bottom:12px}
.var-tab{padding:5px 14px;border-radius:4px;cursor:pointer;font-size:0.78rem;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.04);transition:all .2s}
.var-tab:hover{background:rgba(192,132,252,0.1)}
.var-tab.active{background:rgba(192,132,252,0.2);border-color:#c084fc;color:#c084fc;font-weight:600}

.hm-wrap{overflow-x:auto;margin-bottom:8px}
.hm{display:grid;gap:1px;font-size:0.55rem;background:rgba(0,0,0,0.2);border-radius:4px;overflow:hidden}
.hm-cell{min-height:22px;display:flex;align-items:center;justify-content:center;padding:1px;transition:opacity .15s;cursor:default}
.hm-cell:hover{opacity:0.8;outline:1px solid rgba(255,255,255,0.3)}
.hm-hdr{background:rgba(192,132,252,0.15);color:#c084fc;font-weight:700;font-size:0.58rem}

.xai-grid-2{display:grid;grid-template-columns:3fr 1fr;gap:12px;align-items:start}
.xai-grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.xai-grid-4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px}
.chart-box{background:rgba(255,255,255,0.03);border-radius:8px;padding:10px}
.chart-box canvas{width:100%!important;height:220px!important}
.chart-title{color:#c084fc;font-size:0.82rem;font-weight:600;text-align:center;margin-bottom:6px}

.fi-bar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:0.78rem}
.fi-bar .fi-name{min-width:55px;text-align:right;color:#aaa;font-size:0.72rem}
.fi-bar .fi-fill{height:20px;border-radius:4px;display:flex;align-items:center;padding:0 8px;font-weight:600;font-size:0.7rem;min-width:35px;color:#fff}
.empty{text-align:center;color:#666;padding:40px;font-size:0.9rem}
.note{font-size:0.72rem;color:#888;margin-bottom:8px}

@media(max-width:1100px){.grid{grid-template-columns:1fr}.xai-grid-2{grid-template-columns:1fr}.xai-grid-3{grid-template-columns:1fr}.xai-grid-4{grid-template-columns:1fr 1fr}}
</style></head><body>
<div class="container">
<a class="back-link" href="/">&larr; Back to Main Dashboard</a>
<h1>M10: XAI Dashboard</h1>
<p class="subtitle">Token Attribution Maps &middot; Layer-wise Relevance Propagation &middot; Attention Rollout</p>
<div id="xSt" class="st ok">Loading scenarios...</div>
<div class="grid">
<div class="card">
<h2>Test Scenarios</h2>
<ul class="sc-list" id="xScList"></ul>
</div>
<div class="card">
<div id="xaiContent"><div class="empty">Select a scenario from the list</div></div>
</div>
</div>
</div>

<script>
let xScenarios=[], xCharts={};
const fmt=n=>{const a=Math.abs(n);if(a>=1e9)return(n/1e9).toFixed(2)+'e9';if(a>=1e6)return(n/1e6).toFixed(2)+'e6';if(a>=1e3)return(n/1e3).toFixed(2)+'e3';if(a<0.01&&a>0)return n.toExponential(2);return n.toFixed(2)};

function divergeColor(v,mx){
    const n=Math.min(Math.abs(v)/Math.max(mx,1e-10),1);
    if(v>=0) return 'rgba(255,'+Math.round(80*(1-n))+','+Math.round(80*(1-n))+','+(0.12+n*0.88)+')';
    return 'rgba('+Math.round(80*(1-n))+','+Math.round(130*(1-n))+',255,'+(0.12+n*0.88)+')';
}
function seqColor(v,mx){
    const n=Math.min(v/Math.max(mx,1e-10),1);
    return 'rgba(192,132,252,'+(0.04+n*0.92)+')';
}

function renderLRP(data,varName){
    const el=document.getElementById('lrpSection');
    const lrp=data.lrp_attribution[varName];
    const fnames=data.feature_names;
    const nSteps=lrp.length, nFeat=fnames.length;
    let mx=0;
    for(let s=0;s<nSteps;s++) for(let f=0;f<nFeat;f++) mx=Math.max(mx,Math.abs(lrp[s][f]));
    if(mx<1e-10) mx=1;

    let h='<h3>Feature Attribution Heatmap &mdash; '+varName+' (Gradient &times; Input LRP)</h3>';
    h+='<p class="note">Rows = input features, Columns = degradation steps. Red = positive relevance (feature pushes prediction up), Blue = negative relevance. Hover for values.</p>';
    h+='<div class="hm-wrap">';
    h+='<div class="hm" style="grid-template-columns:65px repeat('+nSteps+',1fr)">';
    h+='<div class="hm-cell hm-hdr"></div>';
    for(let s=0;s<nSteps;s++) h+='<div class="hm-cell hm-hdr">S'+(s+1)+'</div>';
    for(let f=0;f<nFeat;f++){
        h+='<div class="hm-cell hm-hdr">'+fnames[f]+'</div>';
        for(let s=0;s<nSteps;s++){
            const v=lrp[s][f];
            const bg=divergeColor(v,mx);
            h+='<div class="hm-cell" style="background:'+bg+'" title="'+fnames[f]+' → Step '+(s+1)+': '+v.toFixed(5)+'"></div>';
        }
    }
    h+='</div></div>';

    // Feature importance bars
    const fi=data.feature_importance[varName];
    h+='<h3>Feature Importance &mdash; '+varName+'</h3>';
    h+='<p class="note">Percentage of total absolute relevance attributed to each input feature, summed across all 21 steps.</p>';
    const fiArr=fnames.map((name,i)=>({name,pct:fi[i]})).sort((a,b)=>b.pct-a.pct);
    const maxPct=Math.max(...fiArr.map(f=>f.pct),1);
    const barColors=['#c084fc','#a78bfa','#818cf8','#60a5fa','#38bdf8','#2dd4bf','#4ade80','#a3e635'];
    fiArr.forEach((f,i)=>{
        const w=Math.max(f.pct/maxPct*100,8);
        h+='<div class="fi-bar"><span class="fi-name">'+f.name+'</span><div class="fi-fill" style="width:'+w+'%;background:'+barColors[i%8]+'">'+f.pct.toFixed(1)+'%</div></div>';
    });
    el.innerHTML=h;
}

function renderCrossAttn(data){
    const el=document.getElementById('crossSection');
    const iters=data.cross_attention_per_iter;
    const rollup=data.cross_attention_rollout;
    let h='<h3>Cross-Attention: How Each Slot Attends to Input</h3>';
    h+='<p class="note">Each bar shows how strongly a slot queries the input embedding. Higher = more information pulled from the 8 input features into that slot. Rollout averages across all 3 iterations.</p>';
    h+='<div class="xai-grid-4">';
    for(let it=0;it<iters.length;it++){
        h+='<div class="chart-box"><div class="chart-title">Iteration '+(it+1)+'</div><canvas id="xCross'+it+'"></canvas></div>';
    }
    h+='<div class="chart-box"><div class="chart-title">Rollout (Avg)</div><canvas id="xCrossRoll"></canvas></div>';
    h+='</div>';
    el.innerHTML=h;

    const labels=Array.from({length:21},(_,i)=>'S'+(i+1));
    for(let it=0;it<iters.length;it++){
        if(xCharts['xCross'+it]) xCharts['xCross'+it].destroy();
        xCharts['xCross'+it]=new Chart(document.getElementById('xCross'+it),{type:'bar',
            data:{labels,datasets:[{data:iters[it],backgroundColor:'rgba(192,132,252,0.6)',borderColor:'#c084fc',borderWidth:1}]},
            options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
                scales:{y:{grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#aaa',font:{size:9}}},
                        x:{grid:{display:false},ticks:{color:'#aaa',font:{size:7}}}}}
        });
    }
    if(xCharts.xCrossRoll) xCharts.xCrossRoll.destroy();
    xCharts.xCrossRoll=new Chart(document.getElementById('xCrossRoll'),{type:'bar',
        data:{labels,datasets:[{data:rollup,backgroundColor:'rgba(250,204,21,0.6)',borderColor:'#facc15',borderWidth:1}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
            scales:{y:{grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#aaa',font:{size:9}}},
                    x:{grid:{display:false},ticks:{color:'#aaa',font:{size:7}}}}}
    });
}

function renderSelfAttn(data){
    const el=document.getElementById('selfSection');
    const iters=data.self_attention_per_iter;
    const rollup=data.self_attention_rollout;
    const N=21;
    let h='<h3>Self-Attention: Slot-to-Slot Information Flow</h3>';
    h+='<p class="note">Row = query slot, Column = key slot. Brighter = stronger attention weight. Rollout shows aggregated flow across all iterations with residual connection mixing (50% identity, 50% attention).</p>';
    h+='<div class="xai-grid-4">';

    const allMaps=[...iters,rollup];
    const titles=['Iteration 1','Iteration 2','Iteration 3','Rollout'];
    for(let mi=0;mi<allMaps.length;mi++){
        const mat=allMaps[mi];
        let mx=0;
        for(let i=0;i<N;i++) for(let j=0;j<N;j++) mx=Math.max(mx,mat[i][j]);
        if(mx<1e-10) mx=1;
        h+='<div><div class="chart-title">'+titles[mi]+'</div>';
        h+='<div class="hm-wrap">';
        h+='<div class="hm" style="grid-template-columns:repeat('+N+',1fr)">';
        for(let i=0;i<N;i++){
            for(let j=0;j<N;j++){
                const v=mat[i][j];
                const bg=seqColor(v,mx);
                h+='<div class="hm-cell" style="background:'+bg+';min-height:13px;min-width:13px" title="S'+(i+1)+'→S'+(j+1)+': '+v.toFixed(3)+'"></div>';
            }
        }
        h+='</div></div></div>';
    }
    h+='</div>';
    el.innerHTML=h;
}

let currentVar='KL';
function selectVar(vn){
    currentVar=vn;
    document.querySelectorAll('.var-tab').forEach(t=>t.classList.toggle('active',t.dataset.var===vn));
    if(window._xaiData) renderLRP(window._xaiData,vn);
}

async function showXai(idx){
    document.querySelectorAll('.sc-list li').forEach((li,i)=>li.classList.toggle('active',i===idx));
    const el=document.getElementById('xaiContent');
    el.innerHTML='<div class="empty" style="color:#c084fc">&#9881; Computing XAI analysis...</div>';
    try{
        const s=xScenarios[idx];
        const res=await fetch('/api/xai/'+idx);
        const data=await res.json();
        if(data.error){el.innerHTML='<div class="empty" style="color:#ff5757">'+data.error+'</div>';return}
        window._xaiData=data;

        // Params bar
        const pn={'PI':'PI','Gmax':'G_max','v':'ν','Dp':'D_p','Tp':'T_p','Lp':'L_p','Ip':'I_p','Dp_Lp':'D_p/L_p'};
        let h='<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;padding:8px;background:rgba(255,255,255,0.03);border-radius:6px">';
        for(const[k,v] of Object.entries(s.params)) h+='<div style="background:rgba(192,132,252,0.1);border:1px solid rgba(192,132,252,0.2);border-radius:4px;padding:3px 7px;font-size:0.72rem"><span style="color:#888">'+(pn[k]||k)+':</span> <span style="color:#c084fc;font-weight:600">'+fmt(v)+'</span></div>';
        h+='</div>';

        // Variable tabs
        h+='<div class="var-tabs">';
        for(const vn of ['KL','KR','KLR']) h+='<div class="var-tab'+(vn===currentVar?' active':'')+'" data-var="'+vn+'" onclick="selectVar(\''+vn+'\')">'+vn+'</div>';
        h+='</div>';

        // Sections
        h+='<div id="lrpSection" class="card"></div>';
        h+='<div id="crossSection" class="card"></div>';
        h+='<div id="selfSection" class="card"></div>';

        el.innerHTML=h;
        renderLRP(data,currentVar);
        renderCrossAttn(data);
        renderSelfAttn(data);
    }catch(e){el.innerHTML='<div class="empty" style="color:#ff5757">Error: '+e+'</div>'}
}

window.onload=async()=>{
    const st=document.getElementById('xSt');
    try{
        const r=await(await fetch('/api/scenarios')).json();
        if(r.error){st.textContent=r.error;return}
        xScenarios=r.scenarios;
        st.textContent='Model loaded \u2014 '+xScenarios.length+' test scenarios \u2014 Select one for XAI analysis';
        const list=document.getElementById('xScList');
        xScenarios.forEach((s,i)=>{
            const li=document.createElement('li');
            li.innerHTML='<div>'+s.label+'</div><div class="sc-params">PI='+fmt(s.params.PI)+' Gmax='+fmt(s.params.Gmax)+' Lp='+fmt(s.params.Lp)+'</div>';
            li.onclick=()=>showXai(i);
            list.appendChild(li);
        });
        if(xScenarios.length>0) showXai(0);
    }catch(e){st.textContent='Failed: '+e}
};
</script></body></html>'''


@app.route('/xai')
def xai_page():
    return render_template_string(XAI_HTML)


@app.route('/api/xai/<int:scenario_idx>')
def api_xai(scenario_idx):
    if scenarios_cache is None and not load_all():
        return jsonify({'error': 'Model not loaded. Run train.py first.'})
    if scenario_idx < 0 or scenario_idx >= len(scenarios_cache):
        return jsonify({'error': f'Invalid scenario index: {scenario_idx}'})
    analyzer = XAIAnalyzer(model, feature_names)
    x_scaled = torch.FloatTensor(test_data['X_scaled'][scenario_idx:scenario_idx+1])
    xai_result = analyzer.analyze(x_scaled)
    xai_result['scenario'] = scenarios_cache[scenario_idx]
    return jsonify(xai_result)


@app.route('/api/analysis')
def api_analysis():
    if scenarios_cache is None and not load_all():
        return jsonify({'error': 'Model not loaded. Run train.py first.'})
    if scenario_analysis is None:
        return jsonify({'error': 'Run analyze_scenarios.py first.'})
    return jsonify(scenario_analysis)


DIAGNOSIS_HTML = r'''<!DOCTYPE html><html><head><title>M9 Diagnosis - Worst Scenarios</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);min-height:100vh;color:#fff;padding:20px}
.container{max-width:1600px;margin:0 auto}
h1{text-align:center;font-size:1.5rem;margin-bottom:6px;color:#ffa500}
.subtitle{text-align:center;font-size:0.78rem;color:#f0a050;margin-bottom:16px}
h2{color:#ffa500;margin-bottom:10px;font-size:1rem}
h3{color:#00d2ff;margin:14px 0 8px;font-size:0.88rem}
.card{background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;border:1px solid rgba(255,255,255,0.1);margin-bottom:12px}
.back-link{display:inline-block;margin-bottom:14px;color:#00d2ff;text-decoration:underline;font-size:0.85rem}
.back-link:hover{color:#fff}

.banner{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}
.bstat{background:rgba(255,165,0,0.08);border:1px solid rgba(255,165,0,0.25);border-radius:10px;padding:12px 16px;text-align:center}
.bstat .label{font-size:0.68rem;color:#f0a050;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.bstat .value{font-size:1.3rem;font-weight:700;color:#ffa500}
.bstat .sub{font-size:0.66rem;color:#886;margin-top:2px}

table.dtbl{width:100%;border-collapse:collapse;font-size:0.76rem;margin-bottom:12px}
table.dtbl th{background:rgba(255,165,0,0.1);color:#ffa500;padding:8px 12px;text-align:center;position:sticky;top:0}
table.dtbl td{padding:6px 12px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:center}
table.dtbl tr:hover{background:rgba(255,165,0,0.05)}
.err-good{color:#00ff88}.err-mid{color:#ffa500}.err-bad{color:#ff5757}

.fi-bar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:0.78rem}
.fi-bar .fi-name{min-width:60px;text-align:right;color:#aaa}
.fi-bar .fi-fill{height:22px;border-radius:4px;display:flex;align-items:center;padding:0 8px;font-weight:600;font-size:0.72rem;min-width:40px}
.fi-bar .fi-pct{color:#ffa500;min-width:45px}

.diag-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.chart-box{background:rgba(255,255,255,0.03);border-radius:8px;padding:10px}
.chart-box canvas{width:100%!important;height:280px!important}
.chart-title{color:#00d2ff;font-size:0.82rem;font-weight:600;text-align:center;margin-bottom:6px}

.sc-detail{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px}
.sc-detail canvas{width:100%!important;height:200px!important}
.sc-card-title{color:#ffa500;font-size:0.85rem;font-weight:700;margin-bottom:6px}

.tag{display:inline-block;padding:2px 6px;border-radius:3px;font-size:0.68rem;font-weight:600}
.tag-bad{background:rgba(255,87,87,0.15);color:#ff5757;border:1px solid rgba(255,87,87,0.3)}
.tag-ok{background:rgba(0,255,136,0.1);color:#00ff88;border:1px solid rgba(0,255,136,0.2)}

.diag-table{width:100%;border-collapse:collapse;font-size:0.76rem;margin-bottom:12px}
.diag-table th{background:rgba(0,210,255,0.1);color:#00d2ff;padding:6px 10px;text-align:center}
.diag-table td{padding:5px 10px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:center}
.diag-table tr:hover{background:rgba(0,210,255,0.04)}

.sigma-bar{display:inline-block;height:8px;border-radius:4px;vertical-align:middle}
.worst-curves{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
@media(max-width:1100px){.banner{grid-template-columns:repeat(3,1fr)}.diag-grid{grid-template-columns:1fr}.sc-detail{grid-template-columns:1fr}.worst-curves{grid-template-columns:1fr}}
</style></head><body>
<div class="container">
<a class="back-link" href="/">&larr; Back to Main Dashboard</a>
<h1>Worst-Scenario Diagnosis &amp; Feature Importance</h1>
<p class="subtitle">Curve-Based Error Analysis &mdash; M9 SwiGLU &Psi;-NN</p>
<div id="st" class="card" style="text-align:center;color:#888">Loading analysis...</div>
<div id="content" style="display:none">

<!-- Global Banner -->
<div class="banner" id="globalBanner"></div>

<!-- Per-Variable Distribution -->
<div class="card">
<h2>Per-Variable Curve Error Distribution</h2>
<table class="dtbl" id="pvdTable"><thead><tr>
<th>Variable</th><th>MAPE (mean)</th><th>MAPE (median)</th><th>MAPE (p95)</th><th>MAPE (max)</th><th>R² (mean)</th><th>R² (min)</th>
</tr></thead><tbody></tbody></table>
</div>

<!-- Feature Importance -->
<div class="card">
<h2>Feature Importance (% Influence on Prediction Error)</h2>
<div class="diag-grid">
<div id="fiBars"></div>
<div class="chart-box"><div class="chart-title">Feature Importance</div><canvas id="cFI"></canvas></div>
</div>
</div>

<!-- Worst Scenarios Table -->
<div class="card">
<h2>Top 10 Worst Scenarios (by Curve MAPE)</h2>
<table class="dtbl" id="worstTable"><thead><tr>
<th>Rank</th><th>Scenario</th><th>Avg MAPE</th><th>Avg NRMSE</th><th>Avg R²</th><th>KL MAPE</th><th>KR MAPE</th><th>KLR MAPE</th>
</tr></thead><tbody></tbody></table>
</div>

<!-- Worst Scenario Curves -->
<div class="card">
<h2>Worst Scenario Curves (Target vs Predicted)</h2>
<div id="worstCurves"></div>
</div>

<!-- Diagnosis: Worst vs All -->
<div class="card">
<h2>Diagnosis: Why Do Worst Scenarios Fail?</h2>
<p style="font-size:0.78rem;color:#aaa;margin-bottom:10px">Comparing input feature means of the 10 worst scenarios vs the full test set. Large σ-deviations indicate the model is being pushed outside its training comfort zone.</p>
<table class="diag-table" id="diagTable"><thead><tr>
<th>Feature</th><th>Worst Mean</th><th>All Mean</th><th>σ-Deviation</th><th></th>
</tr></thead><tbody></tbody></table>
<div class="card" style="border-color:rgba(255,165,0,0.3);margin-top:10px">
<h3 style="color:#ffa500">Root-Cause Summary</h3>
<ul id="rootCause" style="font-size:0.78rem;color:#ccc;padding-left:18px;line-height:1.7"></ul>
</div>
</div>

</div><!-- content -->
</div><!-- container -->

<script>
const fmt=n=>{const a=Math.abs(n);if(a>=1e9)return(n/1e9).toFixed(2)+'e9';if(a>=1e6)return(n/1e6).toFixed(2)+'e6';if(a>=1e3)return(n/1e3).toFixed(2)+'e3';if(a<0.01&&a>0)return n.toExponential(2);return n.toFixed(2)};
const mapeClass=v=>v<5?'err-good':v<15?'err-mid':'err-bad';
const r2Class=v=>v>0.99?'err-good':v>0.95?'err-mid':'err-bad';
let wCharts={};

function mkChart(canvasId,label,tgt,pred,steps){
    if(wCharts[canvasId])wCharts[canvasId].destroy();
    const ctx=document.getElementById(canvasId);
    wCharts[canvasId]=new Chart(ctx,{type:'line',data:{labels:steps,datasets:[
        {label:'Target',data:tgt,borderColor:'#00ff88',fill:false,tension:.3,pointRadius:2,borderWidth:2},
        {label:'Predicted',data:pred,borderColor:'#ff6b6b',fill:false,tension:.3,pointRadius:2,borderWidth:2,borderDash:[5,3]}
    ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#aaa',font:{size:10}}}},scales:{x:{title:{display:true,text:'Step',color:'#888'},grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#aaa'}},y:{title:{display:true,text:label,color:'#888'},grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#aaa',callback:v=>fmt(v)}}}}});
}

window.onload=async()=>{
    const st=document.getElementById('st');
    try{
        const [aRes, sRes]=await Promise.all([fetch('/api/analysis'),fetch('/api/scenarios')]);
        const data=await aRes.json();
        const sData=await sRes.json();
        if(data.error){st.textContent=data.error;return}
        st.style.display='none';
        document.getElementById('content').style.display='block';

        const gm=data.global_metrics;
        const scenarios=sData.scenarios||[];

        // Global Banner
        document.getElementById('globalBanner').innerHTML=`
            <div class="bstat"><div class="label">Test Scenarios</div><div class="value">${gm.n_test}</div></div>
            <div class="bstat"><div class="label">Mean Curve MAPE</div><div class="value">${gm.avg_curve_mape_pct}%</div></div>
            <div class="bstat"><div class="label">Median MAPE</div><div class="value">${gm.median_curve_mape_pct}%</div></div>
            <div class="bstat"><div class="label">Mean Curve R²</div><div class="value">${gm.avg_curve_r2}</div></div>
            <div class="bstat"><div class="label">P95 MAPE</div><div class="value">${gm.p95_curve_mape_pct}%</div><div class="sub">95th percentile</div></div>`;

        // Per-variable distribution
        const pvdTb=document.querySelector('#pvdTable tbody');
        for(const vn of ['KL','KR','KLR']){
            const p=data.per_variable_distribution[vn];
            pvdTb.innerHTML+=`<tr><td><strong>${vn}</strong></td>
                <td class="${mapeClass(p.mape_mean)}">${p.mape_mean}%</td>
                <td>${p.mape_median}%</td><td>${p.mape_p95}%</td>
                <td class="err-bad">${p.mape_max}%</td>
                <td class="${r2Class(p.r2_mean)}">${p.r2_mean}</td>
                <td class="${r2Class(p.r2_min)}">${p.r2_min}</td></tr>`;
        }

        // Feature Importance
        const ranking=data.feature_ranking||[];
        const maxPct=ranking.length?ranking[0].influence_pct:100;
        const fiColors=['#ff6b6b','#ffa500','#ffd700','#00ff88','#4dc9f6','#c084fc','#f472b6','#aaa'];
        let fiH='';
        ranking.forEach((f,i)=>{
            const w=Math.max(f.influence_pct/maxPct*100,8);
            fiH+=`<div class="fi-bar"><span class="fi-name">${f.feature}</span><div class="fi-fill" style="width:${w}%;background:${fiColors[i]}">${f.influence_pct}%</div></div>`;
        });
        document.getElementById('fiBars').innerHTML=fiH;
        // FI Chart
        new Chart(document.getElementById('cFI'),{type:'doughnut',data:{
            labels:ranking.map(f=>f.feature),
            datasets:[{data:ranking.map(f=>f.influence_pct),backgroundColor:fiColors.slice(0,ranking.length),borderWidth:0}]
        },options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right',labels:{color:'#ccc',font:{size:11}}}}}});

        // Worst Scenarios Table
        const worst=data.worst_scenarios||[];
        const wtb=document.querySelector('#worstTable tbody');
        worst.forEach((s,i)=>{
            const o=s.overall;
            const sid=s.scenario_id-1;
            wtb.innerHTML+=`<tr style="cursor:pointer" onclick="window.location.href='/#scenario=${sid}'" title="Click to view scenario details">
                <td>${i+1}</td><td><a href="/#scenario=${sid}" style="color:#00d2ff;text-decoration:underline">S${s.scenario_id}</a></td>
                <td class="${mapeClass(o.avg_curve_mape_pct)}">${o.avg_curve_mape_pct}%</td>
                <td>${o.avg_curve_nrmse_pct}%</td>
                <td class="${r2Class(o.avg_curve_r2)}">${o.avg_curve_r2.toFixed(4)}</td>
                <td class="${mapeClass(s.per_variable.KL.curve_mape_pct)}">${s.per_variable.KL.curve_mape_pct}%</td>
                <td class="${mapeClass(s.per_variable.KR.curve_mape_pct)}">${s.per_variable.KR.curve_mape_pct}%</td>
                <td class="${mapeClass(s.per_variable.KLR.curve_mape_pct)}">${s.per_variable.KLR.curve_mape_pct}%</td>
            </tr>`;
        });

        // Worst Scenario Curves (top 5)
        const wcDiv=document.getElementById('worstCurves');
        worst.slice(0,5).forEach((ws,wi)=>{
            const sid=ws.scenario_id-1;
            const sc=scenarios.find(s=>s.id===sid);
            if(!sc)return;
            const rank=wi+1;
            let h=`<div class="card" style="border-color:rgba(255,87,87,0.3)">`;
            h+=`<div class="sc-card-title" style="cursor:pointer" onclick="window.location.href='/#scenario=${sid}'">#${rank}: <a href="/#scenario=${sid}" style="color:#00d2ff;text-decoration:underline">Scenario ${ws.scenario_id}</a> &mdash; MAPE=${ws.overall.avg_curve_mape_pct}% &nbsp; R²=${ws.overall.avg_curve_r2.toFixed(4)}</div>`;
            h+=`<div style="font-size:0.72rem;color:#888;margin-bottom:8px">`;
            for(const[k,v]of Object.entries(ws.params))h+=`${k}=${fmt(v)} &nbsp;`;
            h+=`</div>`;
            h+=`<div class="sc-detail">`;
            for(const vn of ['KL','KR','KLR']){
                const cm=ws.per_variable[vn];
                h+=`<div class="chart-box"><div class="chart-title">${vn} <span class="${mapeClass(cm.curve_mape_pct)}">(MAPE ${cm.curve_mape_pct}%)</span></div><canvas id="wc${rank}_${vn}"></canvas></div>`;
            }
            h+=`</div></div>`;
            wcDiv.innerHTML+=h;
            setTimeout(()=>{
                for(const vn of ['KL','KR','KLR']){
                    mkChart(`wc${rank}_${vn}`,vn,sc.target[vn],sc.predicted[vn],sc.steps);
                }
            },50);
        });

        // Diagnosis Table
        const diag=data.diagnosis||{};
        const diagTb=document.querySelector('#diagTable tbody');
        const featureOrder=['PI','Gmax','v','Dp','Tp','Lp','Ip','Dp_Lp'];
        featureOrder.forEach(col=>{
            const d=diag[col];if(!d)return;
            const absD=Math.abs(d.deviation_sigma);
            const barW=Math.min(absD/1.5*100,100);
            const barColor=absD>0.5?'#ff5757':absD>0.3?'#ffa500':'#00ff88';
            const sign=d.deviation_sigma>0?'+':'';
            diagTb.innerHTML+=`<tr>
                <td><strong>${col}</strong></td>
                <td>${d.worst_mean.toFixed(4)}</td>
                <td>${d.all_mean.toFixed(4)}</td>
                <td style="color:${barColor}">${sign}${d.deviation_sigma.toFixed(2)}σ</td>
                <td style="width:120px"><div class="sigma-bar" style="width:${barW}%;background:${barColor}"></div></td>
            </tr>`;
        });

        // Root Cause
        const rootCause=document.getElementById('rootCause');
        const sorted=Object.entries(diag).sort((a,b)=>Math.abs(b[1].deviation_sigma)-Math.abs(a[1].deviation_sigma));
        const explanations={
            'PI':'The Plasticity Index controls clay cyclic softening. Extreme PI values produce highly nonlinear degradation that the 5 prototypes struggle to represent.',
            'Gmax':'Maximum shear modulus drives absolute stiffness magnitude. Extreme values amplify absolute errors and push the model into extrapolation.',
            'v':'Poisson ratio governs volumetric vs shear response. Near-incompressible soils produce very different degradation signatures.',
            'Dp':'Pile diameter is constant across scenarios (no variation), so it has zero influence on error differences.',
            'Tp':'Pile wall thickness affects structural rigidity. Unusual thickness creates atypical stiffness evolution patterns.',
            'Lp':'Pile length controls load transfer depth. Longer piles engage more soil layers, creating complex multi-mode degradation.',
            'Ip':'Moment of inertia determines bending stiffness. Extreme values lead to degradation dynamics the prototypes cannot fully capture.',
            'Dp_Lp':'Slenderness ratio is a key dimensionless design parameter. Very slender vs short stiff piles behave fundamentally differently.'
        };
        sorted.slice(0,4).forEach(([col,d])=>{
            if(Math.abs(d.deviation_sigma)<0.05)return;
            const dir=d.deviation_sigma>0?'higher':'lower';
            rootCause.innerHTML+=`<li><strong>${col}</strong>: worst scenarios are ${Math.abs(d.deviation_sigma).toFixed(2)}σ ${dir} than average. ${explanations[col]||''}</li>`;
        });
        rootCause.innerHTML+=`<li><strong>Dynamic range</strong>: worst scenarios have ~2× larger target curve ranges, making relative errors more sensitive to absolute offsets.</li>`;
        rootCause.innerHTML+=`<li><strong>Extreme combinations</strong>: multiple features simultaneously at distribution tails compound extrapolation difficulty.</li>`;

    }catch(e){st.textContent='Failed: '+e;st.style.color='#ff5757'}
};
</script></body></html>'''


if __name__ == '__main__':
    print("=" * 50)
    print("M10: XAI-Enhanced SwiGLU Psi-NN")
    print("=" * 50)
    if load_all():
        k = psi_config.get('num_prototypes', '?') if psi_config else '?'
        print(f"Psi-Model loaded | k*={k} prototypes | {len(scenarios_cache)} scenarios | {max_seq_len} steps")
        if scenario_analysis:
            gm = scenario_analysis.get('global_metrics', {})
            print(f"Analysis loaded | MAPE={gm.get('avg_curve_mape_pct','?')}% | R²={gm.get('avg_curve_r2','?')}")
    else:
        print("Run: python train.py first")
    print("Server: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=True)
