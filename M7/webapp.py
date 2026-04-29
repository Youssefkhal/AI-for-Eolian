"""
M7: Ψ-NN Pile Stiffness Degradation - Web Application
=====================================================
Shows Ψ-model predictions alongside structure discovery:
  - Prototype clustering, relation matrix, compression stats
  - Comparison with M6 teacher
  - Per-slot, per-variable accuracy metrics
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


class SlotAttentionPsiModel(nn.Module):
    """Ψ-NN structured model (must match train.py exactly)."""

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
            self.register_buffer('relation_matrix', torch.FloatTensor(relation_matrix))
        else:
            R = np.zeros((self.num_drop_slots, num_prototypes))
            slots_per_proto = self.num_drop_slots // num_prototypes
            for p in range(num_prototypes):
                start = p * slots_per_proto
                end = start + slots_per_proto if p < num_prototypes - 1 else self.num_drop_slots
                R[start:end, p] = 1.0
            self.register_buffer('relation_matrix', torch.FloatTensor(R))

        self.slot_scales = nn.Parameter(torch.ones(self.num_drop_slots, 1))

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

    def reconstruct_drop_slots(self, B):
        protos = self.prototype_slots.expand(B, -1, -1)
        drop_slots = torch.matmul(self.relation_matrix, protos)
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
psi_config, psi_discovery, comparison_data = None, None, None
scenarios_cache, metrics_cache = None, None


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
    global psi_config, psi_discovery, comparison_data
    global scenarios_cache, metrics_cache
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

        m_overall = calc_metrics(Y_original, pred_original)
        m_per_var = {}
        for vi, vn in enumerate(var_names):
            m_per_var[vn] = calc_metrics(Y_original[:, :, vi], pred_original[:, :, vi])

        m_per_slot = []
        for s in range(max_seq_len):
            sm = calc_metrics(Y_original[:, s, :], pred_original[:, s, :])
            sm['per_variable'] = {}
            for vi, vn in enumerate(var_names):
                sm['per_variable'][vn] = calc_metrics(
                    Y_original[:, s, vi:vi + 1], pred_original[:, s, vi:vi + 1])
            sm['slot'] = s + 1
            sm['type'] = 'initial' if s == 0 else 'drop'
            m_per_slot.append(sm)

        metrics_cache = {
            'overall': m_overall,
            'per_variable': m_per_var,
            'per_slot': m_per_slot,
        }

        scenarios_cache = []
        for i in range(n_test):
            params = {col: float(X_original[i, j]) for j, col in enumerate(input_cols)}
            y_t, y_p = Y_original[i], pred_original[i]
            sc_metrics = calc_metrics(y_t, y_p)
            sc_var_metrics = {}
            for vi, vn in enumerate(var_names):
                sc_var_metrics[vn] = calc_metrics(y_t[:, vi], y_p[:, vi])
            scenarios_cache.append({
                'id': i, 'params': params,
                'label': f"Scenario {i + 1}",
                'target': {vn: Y_original[i, :, vi].tolist() for vi, vn in enumerate(var_names)},
                'predicted': {vn: pred_original[i, :, vi].tolist() for vi, vn in enumerate(var_names)},
                'steps': list(range(1, max_seq_len + 1)),
                'metrics': sc_metrics, 'metrics_per_var': sc_var_metrics,
            })
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return False


HTML = '''<!DOCTYPE html><html><head><title>M7 Psi-NN Pile Stiffness</title>
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
<h1>M7: &Psi;-NN Pile Stiffness Degradation</h1>
<p class="subtitle">Structure Discovery &amp; Compression via Knowledge Distillation (Liu et al., 2025)</p>
<div id="st" class="st ok">Loading model...</div>

<div id="psiSection"></div>
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
<h2>Per-Slot Accuracy (Across All Test Scenarios)</h2>
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
        h+='<table class="comp-table"><thead><tr><th>Model</th><th>Params</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th>';
        for(const vn of ['KL','KR','KLR'])h+=`<th>${vn} R&sup2;</th>`;
        h+='</tr></thead><tbody>';
        const models=[['M6 Teacher',comp.teacher_m6],['Stage-A Student',comp.student_stage_a],['&Psi;-Model (M7)',comp.psi_model_m7]];
        const bestR2=Math.max(...models.map(m=>m[1].overall.r2));
        for(const [name,m] of models){
            const isBest=m.overall.r2===bestR2;
            h+=`<tr><td><strong>${name}</strong></td><td>${m.params.toLocaleString()}</td>`;
            h+=`<td class="${isBest?'comp-best':''}">${fmtM(m.overall.r2)}</td>`;
            h+=`<td>${fmtS(m.overall.rmse)}</td><td>${fmtS(m.overall.mae)}</td>`;
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

function showScenario(idx){
    document.querySelectorAll('.sc-list li').forEach((li,i)=>li.classList.toggle('active',i===idx));
    const s=scenarios[idx];
    const pn={'PI':'PI','Gmax':'Gmax (Pa)','v':'v','Dp':'Dp (m)','Tp':'Tp (m)','Lp':'Lp (m)','Ip':'Ip','Dp_Lp':'Dp/Lp'};
    let h='<div class="params-bar">';
    for(const[k,v]of Object.entries(s.params))h+=`<div class="param-chip"><span class="lbl">${pn[k]||k}:</span> <span class="val">${fmt(v)}</span></div>`;
    h+='</div>';
    h+='<div class="sc-metrics-bar">';
    h+=`<div class="sc-metric"><span class="ml">R&sup2; = </span><span class="mv">${fmtM(s.metrics.r2)}</span></div>`;
    h+=`<div class="sc-metric"><span class="ml">RMSE = </span><span class="mv">${fmtS(s.metrics.rmse)}</span></div>`;
    h+=`<div class="sc-metric"><span class="ml">MAE = </span><span class="mv">${fmtS(s.metrics.mae)}</span></div>`;
    h+='</div>';
    h+='<div class="sc-var-row">';
    for(const vn of ['KL','KR','KLR']){const m=s.metrics_per_var[vn];h+=`<div class="sc-var-chip"><strong>${vn}</strong> &nbsp; R&sup2;=${fmtM(m.r2)} &nbsp; RMSE=${fmtS(m.rmse)} &nbsp; MAE=${fmtS(m.mae)}</div>`;}
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

function renderOverall(m){
    document.getElementById('overallMetrics').innerHTML=`
        <div class="metric-big"><div class="label">R&sup2;</div><div class="value">${fmtM(m.r2)}</div><div class="sub">Coefficient of Determination</div></div>
        <div class="metric-big"><div class="label">RMSE</div><div class="value">${fmtS(m.rmse)}</div><div class="sub">Root Mean Squared Error</div></div>
        <div class="metric-big"><div class="label">MAE</div><div class="value">${fmtS(m.mae)}</div><div class="sub">Mean Absolute Error</div></div>`;
}

function renderVarMetrics(mv){
    let h='';const colors={'KL':'#00ff88','KR':'#4dc9f6','KLR':'#f67019'};
    for(const vn of ['KL','KR','KLR']){const m=mv[vn];h+=`<div class="var-card"><div class="vname" style="color:${colors[vn]}">${vn}</div><div class="vrow"><span class="vl">R&sup2;</span><span class="vv">${fmtM(m.r2)}</span></div><div class="vrow"><span class="vl">RMSE</span><span class="vv">${fmtS(m.rmse)}</span></div><div class="vrow"><span class="vl">MAE</span><span class="vv">${fmtS(m.mae)}</span></div></div>`;}
    document.getElementById('varMetrics').innerHTML=h;
}

function renderSlotMetrics(slots){
    let h='<table class="slot-tbl"><thead><tr><th>Slot</th><th>Type</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th>';
    for(const vn of ['KL','KR','KLR'])h+=`<th>${vn} R&sup2;</th><th>${vn} RMSE</th><th>${vn} MAE</th>`;
    h+='</tr></thead><tbody>';
    for(const s of slots){
        h+=`<tr><td>${s.slot}</td><td>${s.type==='initial'?'Initial':'Drop '+(s.slot-1)}</td>`;
        h+=`<td>${fmtM(s.r2)}</td><td>${fmtS(s.rmse)}</td><td>${fmtS(s.mae)}</td>`;
        for(const vn of ['KL','KR','KLR']){h+=`<td>${fmtM(s.per_variable[vn].r2)}</td><td>${fmtS(s.per_variable[vn].rmse)}</td><td>${fmtS(s.per_variable[vn].mae)}</td>`;}
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
        renderOverall(metrics.overall);
        renderVarMetrics(metrics.per_variable);
        renderSlotMetrics(metrics.per_slot);
        const list=document.getElementById('scList');
        scenarios.forEach((s,i)=>{
            const li=document.createElement('li');
            li.innerHTML=`<div>${s.label}</div><div class="sc-params">PI=${fmt(s.params.PI)} Gmax=${fmt(s.params.Gmax)} Lp=${fmt(s.params.Lp)}</div><div class="sc-r2">R&sup2;=${fmtM(s.metrics.r2)}</div>`;
            li.onclick=()=>showScenario(i);
            list.appendChild(li);
        });
        if(scenarios.length>0)showScenario(0);
    }catch(e){st.textContent='Failed: '+e;st.className='st err'}
};
</script></body></html>'''


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/scenarios')
def api_scenarios():
    if scenarios_cache is None and not load_all():
        return jsonify({'error': 'Model not loaded. Run train.py first.'})
    psi = {}
    if psi_discovery:
        psi['discovery'] = psi_discovery
    if comparison_data:
        psi['comparison'] = comparison_data
    return jsonify({
        'scenarios': scenarios_cache,
        'metrics': metrics_cache,
        'num_steps': max_seq_len,
        'psi': psi,
    })


if __name__ == '__main__':
    print("=" * 50)
    print("M7: Psi-NN Pile Stiffness Degradation")
    print("=" * 50)
    if load_all():
        k = psi_config.get('num_prototypes', '?') if psi_config else '?'
        print(f"Psi-Model loaded | k*={k} prototypes | {len(scenarios_cache)} scenarios | {max_seq_len} steps")
    else:
        print("Run: python train.py first")
    print("Server: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=True)
