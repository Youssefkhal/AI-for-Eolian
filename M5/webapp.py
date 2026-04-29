"""
Pile Stiffness Degradation - Web Application (M5)
21 Steps with comprehensive accuracy metrics:
  R-squared, Root Mean Squared Error, Mean Absolute Error
  Per system, per variable, per slot, per scenario
"""

from flask import Flask, render_template_string, jsonify
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

NUM_STEPS = 21


class SlotAttentionDegradation(nn.Module):
    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21,
                 max_seq_len=21, dropout=0.1, num_iterations=2):
        super().__init__()
        self.num_slots, self.d_model, self.max_seq_len = num_slots, d_model, max_seq_len
        self.num_iterations = num_iterations
        self.input_embed = nn.Sequential(nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.mlp_norm = nn.LayerNorm(d_model)
        self.seq_decoder = nn.LSTM(d_model, d_model, 2, batch_first=True, dropout=dropout)
        self.decoder_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.decoder_cross_norm = nn.LayerNorm(d_model)
        self.h0_proj = nn.Linear(d_model, d_model * 2)
        self.c0_proj = nn.Linear(d_model, d_model * 2)
        self.initial_proj = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.drop_proj = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3))
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

    def forward(self, x, seq_len=None):
        seq_len = seq_len or self.max_seq_len
        batch_size = x.size(0)
        x_embed = self.input_embed(x).unsqueeze(1)
        slots = self.slots.expand(batch_size, -1, -1)
        for _ in range(self.num_iterations):
            cross_out, _ = self.cross_attn(slots, x_embed, x_embed)
            slots = self.cross_norm(slots + cross_out)
            self_out, _ = self.self_attn(slots, slots, slots)
            slots = self.self_norm(slots + self_out)
            slots = self.mlp_norm(slots + self.slot_mlp(slots))
        slot_initial = slots[:, 0:1, :]
        slot_drops = slots[:, 1:, :]
        initial = self.initial_proj(slot_initial)
        init_vec = slot_initial.squeeze(1)
        h0 = self.h0_proj(init_vec).view(batch_size, 2, self.d_model).permute(1, 0, 2).contiguous()
        c0 = self.c0_proj(init_vec).view(batch_size, 2, self.d_model).permute(1, 0, 2).contiguous()
        drop_agg = slot_drops.mean(dim=1, keepdim=True).expand(-1, seq_len - 1, -1)
        dec_in = drop_agg + self.pos_embed[:, :seq_len - 1, :]
        lstm_out, _ = self.seq_decoder(dec_in, (h0, c0))
        refined = self.decoder_cross_attn(lstm_out, slot_drops, slot_drops)[0]
        refined = self.decoder_cross_norm(lstm_out + refined)
        raw_drops = self.drop_proj(refined)
        drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
        drops_klr = raw_drops[:, :, 2:3]
        drops = torch.cat([drops_kl_kr, drops_klr], dim=2)
        return torch.cat([initial, initial + torch.cumsum(drops, dim=1)], dim=1)


model, scaler_X, scaler_Y, feature_names, max_seq_len, test_data = None, None, None, None, None, None
scenarios_cache = None
metrics_cache = None


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
    global scenarios_cache, metrics_cache
    try:
        scaler_X = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
        scaler_Y = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
        feature_names = joblib.load(os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
        max_seq_len = joblib.load(os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
        test_data = joblib.load(os.path.join(SCRIPT_DIR, 'test_data.pkl'))

        model = SlotAttentionDegradation(len(feature_names), 64, 4, max_seq_len, max_seq_len, 0.1, 2)
        model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'pile_model.pth'), map_location='cpu'))
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

        # Global metrics
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

        # Build scenarios
        scenarios_cache = []
        for i in range(n_test):
            params = {col: float(X_original[i, j]) for j, col in enumerate(input_cols)}

            y_t = Y_original[i]  # (21, 3)
            y_p = pred_original[i]  # (21, 3)

            sc_metrics = calc_metrics(y_t, y_p)
            sc_var_metrics = {}
            for vi, vn in enumerate(var_names):
                sc_var_metrics[vn] = calc_metrics(y_t[:, vi], y_p[:, vi])

            scenarios_cache.append({
                'id': i,
                'params': params,
                'label': f"Scenario {i + 1}",
                'target': {vn: Y_original[i, :, vi].tolist() for vi, vn in enumerate(var_names)},
                'predicted': {vn: pred_original[i, :, vi].tolist() for vi, vn in enumerate(var_names)},
                'steps': list(range(1, max_seq_len + 1)),
                'metrics': sc_metrics,
                'metrics_per_var': sc_var_metrics,
            })
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return False


HTML = '''<!DOCTYPE html><html><head><title>Pile Stiffness - 21 Step Prediction</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);min-height:100vh;color:#fff;padding:20px}
.container{max-width:1600px;margin:0 auto}
h1{text-align:center;font-size:1.5rem;margin-bottom:16px;color:#00d2ff}
h2{color:#00d2ff;margin-bottom:10px;font-size:0.95rem}
h3{color:#00d2ff;margin:14px 0 8px;font-size:0.88rem}
.card{background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;border:1px solid rgba(255,255,255,0.1);margin-bottom:12px}

/* Metrics banner */
.metrics-banner{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.metric-big{flex:1;min-width:200px;background:rgba(0,210,255,0.08);border:1px solid rgba(0,210,255,0.25);border-radius:10px;padding:14px 18px;text-align:center}
.metric-big .label{font-size:0.72rem;color:#88c8e8;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.metric-big .value{font-size:1.5rem;font-weight:700;color:#00d2ff}
.metric-big .sub{font-size:0.7rem;color:#668;margin-top:2px}

/* Variable metrics */
.var-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
.var-card{background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;border:1px solid rgba(255,255,255,0.08)}
.var-card .vname{font-weight:700;color:#00ff88;margin-bottom:6px;font-size:0.9rem}
.var-card .vrow{display:flex;justify-content:space-between;font-size:0.75rem;padding:2px 0}
.var-card .vrow .vl{color:#888}.var-card .vrow .vv{color:#fff;font-weight:600}

/* Grid */
.grid{display:grid;grid-template-columns:260px 1fr;gap:12px}
.sc-list{list-style:none;max-height:60vh;overflow-y:auto}
.sc-list li{padding:8px 10px;border-radius:6px;cursor:pointer;margin-bottom:3px;font-size:0.78rem;border:1px solid transparent;transition:all .2s}
.sc-list li:hover{background:rgba(0,210,255,0.1);border-color:rgba(0,210,255,0.3)}
.sc-list li.active{background:rgba(0,210,255,0.2);border-color:#00d2ff;color:#00d2ff;font-weight:600}
.sc-params{font-size:0.65rem;color:#888;margin-top:2px}
.sc-r2{font-size:0.65rem;color:#00ff88;margin-top:1px}

/* Scenario detail */
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

/* Step table */
.table-wrap{max-height:400px;overflow:auto;margin-bottom:12px}
table.step-tbl{width:100%;border-collapse:collapse;font-size:0.72rem}
table.step-tbl th{background:rgba(0,210,255,0.1);color:#00d2ff;position:sticky;top:0;padding:6px 8px;text-align:right;white-space:nowrap}
table.step-tbl th:first-child,table.step-tbl th:nth-child(2){text-align:center}
table.step-tbl td{padding:5px 8px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:right;white-space:nowrap}
table.step-tbl td:first-child,table.step-tbl td:nth-child(2){text-align:center}
table.step-tbl tr:hover{background:rgba(0,210,255,0.05)}
.err-good{color:#00ff88}.err-mid{color:#ffa500}.err-bad{color:#ff5757}

/* Slot metrics section */
.slot-section{margin-top:16px}
table.slot-tbl{width:100%;border-collapse:collapse;font-size:0.73rem}
table.slot-tbl th{background:rgba(0,210,255,0.1);color:#00d2ff;padding:6px 10px;text-align:center;position:sticky;top:0}
table.slot-tbl td{padding:5px 10px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:center}
table.slot-tbl tr:hover{background:rgba(0,210,255,0.05)}

.st{padding:6px 12px;border-radius:4px;margin-bottom:12px;text-align:center;font-size:0.8rem}
.st.ok{background:rgba(0,255,136,0.1);color:#00ff88}
.st.err{background:rgba(255,87,87,0.1);color:#ff5757}
.empty{text-align:center;color:#666;padding:40px;font-size:0.9rem}
@media(max-width:1100px){.grid{grid-template-columns:1fr}.charts{grid-template-columns:1fr}.var-metrics{grid-template-columns:1fr}}
</style></head><body>
<div class="container">
<h1>Pile Stiffness Degradation &mdash; 21-Slot Prediction</h1>
<div id="st" class="st ok">Loading model...</div>

<!-- Overall metrics -->
<div class="metrics-banner" id="overallMetrics"></div>

<!-- Per-variable metrics -->
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

<!-- Per-slot metrics -->
<div class="card slot-section" id="slotSection">
<h2>Per-Slot Accuracy (Across All Test Scenarios)</h2>
<div class="table-wrap" id="slotTable"></div>
</div>
</div>

<script>
let scenarios=[], metrics={}, charts={};
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

function showScenario(idx){
    document.querySelectorAll('.sc-list li').forEach((li,i)=>li.classList.toggle('active',i===idx));
    const s=scenarios[idx];
    const pn={'PI':'PI','Gmax':'Gmax (Pa)','v':'v','Dp':'Dp (m)','Tp':'Tp (m)','Lp':'Lp (m)','Ip':'Ip','Dp_Lp':'Dp/Lp'};
    let h='<div class="params-bar">';
    for(const[k,v]of Object.entries(s.params))h+=`<div class="param-chip"><span class="lbl">${pn[k]||k}:</span> <span class="val">${fmt(v)}</span></div>`;
    h+='</div>';

    // Scenario metrics
    h+='<div class="sc-metrics-bar">';
    h+=`<div class="sc-metric"><span class="ml">R&sup2; = </span><span class="mv">${fmtM(s.metrics.r2)}</span></div>`;
    h+=`<div class="sc-metric"><span class="ml">RMSE = </span><span class="mv">${fmtS(s.metrics.rmse)}</span></div>`;
    h+=`<div class="sc-metric"><span class="ml">MAE = </span><span class="mv">${fmtS(s.metrics.mae)}</span></div>`;
    h+='</div>';

    // Per-var metrics for scenario
    h+='<div class="sc-var-row">';
    for(const vn of ['KL','KR','KLR']){
        const m=s.metrics_per_var[vn];
        h+=`<div class="sc-var-chip"><strong>${vn}</strong> &nbsp; R&sup2;=${fmtM(m.r2)} &nbsp; RMSE=${fmtS(m.rmse)} &nbsp; MAE=${fmtS(m.mae)}</div>`;
    }
    h+='</div>';

    h+='<div class="legend-bar"><div class="legend-item"><div class="legend-dot" style="background:#00ff88"></div>Target (Real)</div><div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div>Predicted</div></div>';

    h+='<div class="charts">';
    h+='<div class="chc"><div class="cht">KL &mdash; Lateral Stiffness</div><canvas id="cKL"></canvas></div>';
    h+='<div class="chc"><div class="cht">KR &mdash; Rotational Stiffness</div><canvas id="cKR"></canvas></div>';
    h+='<div class="chc"><div class="cht">KLR &mdash; Cross-Coupling</div><canvas id="cKLR"></canvas></div>';
    h+='</div>';

    // Per-step table
    h+='<h3>Per-Step Predicted vs Target Values</h3>';
    h+='<div class="table-wrap"><table class="step-tbl"><thead><tr>';
    h+='<th>Step</th><th>Type</th>';
    for(const vn of ['KL','KR','KLR'])h+=`<th>${vn} Target</th><th>${vn} Pred</th><th>${vn} Err%</th>`;
    h+='</tr></thead><tbody>';
    for(let st=0;st<s.steps.length;st++){
        const tp=st===0?'Initial':'Drop '+(st);
        h+='<tr><td>'+s.steps[st]+'</td><td>'+tp+'</td>';
        for(const vn of ['KL','KR','KLR']){
            const t=s.target[vn][st],p=s.predicted[vn][st];
            const e=errPct(t,p),ec=errClass(e);
            h+=`<td>${fmt(t)}</td><td>${fmt(p)}</td><td class="${ec}">${e}%</td>`;
        }
        h+='</tr>';
    }
    h+='</tbody></table></div>';

    document.getElementById('detail').innerHTML=h;
    mkChart('cKL','KL',s.target.KL,s.predicted.KL,s.steps);
    mkChart('cKR','KR',s.target.KR,s.predicted.KR,s.steps);
    mkChart('cKLR','KLR',s.target.KLR,s.predicted.KLR,s.steps);
}

function renderOverall(m){
    const el=document.getElementById('overallMetrics');
    el.innerHTML=`
        <div class="metric-big"><div class="label">R-squared (R&sup2;)</div><div class="value">${fmtM(m.r2)}</div><div class="sub">Coefficient of Determination</div></div>
        <div class="metric-big"><div class="label">RMSE</div><div class="value">${fmtS(m.rmse)}</div><div class="sub">Root Mean Squared Error</div></div>
        <div class="metric-big"><div class="label">MAE</div><div class="value">${fmtS(m.mae)}</div><div class="sub">Mean Absolute Error</div></div>`;
}

function renderVarMetrics(mv){
    const el=document.getElementById('varMetrics');
    let h='';
    const colors={'KL':'#00ff88','KR':'#4dc9f6','KLR':'#f67019'};
    for(const vn of ['KL','KR','KLR']){
        const m=mv[vn];
        h+=`<div class="var-card"><div class="vname" style="color:${colors[vn]}">${vn}</div>
            <div class="vrow"><span class="vl">R&sup2;</span><span class="vv">${fmtM(m.r2)}</span></div>
            <div class="vrow"><span class="vl">RMSE</span><span class="vv">${fmtS(m.rmse)}</span></div>
            <div class="vrow"><span class="vl">MAE</span><span class="vv">${fmtS(m.mae)}</span></div></div>`;
    }
    el.innerHTML=h;
}

function renderSlotMetrics(slots){
    let h='<table class="slot-tbl"><thead><tr><th>Slot</th><th>Type</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th>';
    for(const vn of ['KL','KR','KLR'])h+=`<th>${vn} R&sup2;</th>`;
    h+='</tr></thead><tbody>';
    for(const s of slots){
        h+=`<tr><td>${s.slot}</td><td>${s.type==='initial'?'Initial':'Drop '+(s.slot-1)}</td>`;
        h+=`<td>${fmtM(s.r2)}</td><td>${fmtS(s.rmse)}</td><td>${fmtS(s.mae)}</td>`;
        for(const vn of ['KL','KR','KLR'])h+=`<td>${fmtM(s.per_variable[vn].r2)}</td>`;
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
        scenarios=r.scenarios;metrics=r.metrics;
        st.textContent=`Model loaded — ${scenarios.length} test scenarios — ${r.num_steps} steps per scenario`;

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
    }catch(e){st.textContent='Failed to load: '+e;st.className='st err'}
};
</script></body></html>'''


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/scenarios')
def api_scenarios():
    if scenarios_cache is None and not load_all():
        return jsonify({'error': 'Model or test data not loaded. Run train.py first.'})
    return jsonify({
        'scenarios': scenarios_cache,
        'metrics': metrics_cache,
        'num_steps': max_seq_len,
    })


if __name__ == '__main__':
    print("=" * 40)
    if load_all():
        print(f"Model loaded | {len(scenarios_cache)} test scenarios | {max_seq_len} steps")
    else:
        print("Run: python train.py first")
    print("Server: http://127.0.0.1:5000")
    print("=" * 40)
    app.run(host='127.0.0.1', port=5000, debug=True)
