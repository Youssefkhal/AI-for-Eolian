"""
Pile Stiffness Degradation - Web Application
Shows target vs predicted values for test scenarios after training.
"""

from flask import Flask, render_template_string, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


class SlotAttentionDegradation(nn.Module):
    def __init__(self, input_size=8, d_model=64, num_heads=4, num_slots=21, max_seq_len=50, dropout=0.1, num_iterations=2):
        super().__init__()
        self.num_slots, self.d_model, self.max_seq_len = num_slots, d_model, max_seq_len
        self.num_iterations = num_iterations
        self.input_embed = nn.Sequential(nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.slot_mlp = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*2, d_model))
        self.mlp_norm = nn.LayerNorm(d_model)
        self.seq_decoder = nn.LSTM(d_model, d_model, 2, batch_first=True, dropout=dropout)
        self.decoder_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.decoder_cross_norm = nn.LayerNorm(d_model)
        self.h0_proj = nn.Linear(d_model, d_model * 2)
        self.c0_proj = nn.Linear(d_model, d_model * 2)
        self.initial_proj = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 3))
        self.drop_proj = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 3))
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
        drop_agg = slot_drops.mean(dim=1, keepdim=True).expand(-1, seq_len-1, -1)
        dec_in = drop_agg + self.pos_embed[:, :seq_len-1, :]
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


def inverse_transform(scaled, scaler_Y):
    """Inverse: unscale -> inverse log1p -> original values."""
    flat = scaled.reshape(-1, 3)
    log_vals = scaler_Y.inverse_transform(flat)
    orig = np.sign(log_vals) * np.expm1(np.abs(log_vals))
    return orig.reshape(scaled.shape)


def load_all():
    global model, scaler_X, scaler_Y, feature_names, max_seq_len, test_data, scenarios_cache
    try:
        scaler_X = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_X.pkl'))
        scaler_Y = joblib.load(os.path.join(SCRIPT_DIR, 'scaler_y.pkl'))
        feature_names = joblib.load(os.path.join(SCRIPT_DIR, 'feature_names.pkl'))
        max_seq_len = joblib.load(os.path.join(SCRIPT_DIR, 'max_seq_len.pkl'))
        test_data = joblib.load(os.path.join(SCRIPT_DIR, 'test_data.pkl'))

        model = SlotAttentionDegradation(len(feature_names), 64, 4, 21, max_seq_len, 0.1, 2)
        model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'pile_model.pth'), map_location='cpu'))
        model.eval()

        # Pre-compute all predictions
        X_scaled = test_data['X_scaled']
        X_original = test_data['X_original']
        Y_original = test_data['Y_original']
        input_cols = test_data['input_cols']

        with torch.no_grad():
            pred_scaled = model(torch.FloatTensor(X_scaled), max_seq_len).numpy()
        pred_original = inverse_transform(pred_scaled, scaler_Y)

        scenarios_cache = []
        for i in range(len(X_original)):
            params = {col: float(X_original[i, j]) for j, col in enumerate(input_cols)}
            scenarios_cache.append({
                'id': i,
                'params': params,
                'label': f"Scenario {i+1}",
                'target': {
                    'KL': Y_original[i, :, 0].tolist(),
                    'KR': Y_original[i, :, 1].tolist(),
                    'KLR': Y_original[i, :, 2].tolist(),
                },
                'predicted': {
                    'KL': pred_original[i, :, 0].tolist(),
                    'KR': pred_original[i, :, 1].tolist(),
                    'KLR': pred_original[i, :, 2].tolist(),
                },
                'steps': list(range(1, max_seq_len + 1)),
            })
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        return False


HTML = '''<!DOCTYPE html><html><head><title>Pile Stiffness - Target vs Predicted</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);min-height:100vh;color:#fff;padding:20px}
.container{max-width:1500px;margin:0 auto}
h1{text-align:center;font-size:1.6rem;margin-bottom:20px;color:#00d2ff}
.grid{display:grid;grid-template-columns:280px 1fr;gap:15px}
.card{background:rgba(255,255,255,0.05);border-radius:10px;padding:15px;border:1px solid rgba(255,255,255,0.1)}
h2{color:#00d2ff;margin-bottom:10px;font-size:0.95rem}
.sc-list{list-style:none;max-height:70vh;overflow-y:auto}
.sc-list li{padding:10px 12px;border-radius:6px;cursor:pointer;margin-bottom:4px;font-size:0.82rem;border:1px solid transparent;transition:all .2s}
.sc-list li:hover{background:rgba(0,210,255,0.1);border-color:rgba(0,210,255,0.3)}
.sc-list li.active{background:rgba(0,210,255,0.2);border-color:#00d2ff;color:#00d2ff;font-weight:600}
.sc-params{font-size:0.7rem;color:#888;margin-top:3px}
.params-bar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;padding:10px;background:rgba(255,255,255,0.03);border-radius:6px}
.param-chip{background:rgba(0,210,255,0.1);border:1px solid rgba(0,210,255,0.2);border-radius:4px;padding:4px 8px;font-size:0.75rem}
.param-chip .lbl{color:#888}.param-chip .val{color:#00d2ff;font-weight:600}
.charts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.chc{background:rgba(255,255,255,0.03);border-radius:8px;padding:12px}
.cht{color:#00d2ff;margin-bottom:8px;font-size:0.85rem;font-weight:600;text-align:center}
canvas{width:100%!important;height:280px!important}
.legend-bar{display:flex;justify-content:center;gap:20px;margin-bottom:12px}
.legend-item{display:flex;align-items:center;gap:6px;font-size:0.8rem}
.legend-dot{width:12px;height:3px;border-radius:2px}
.st{padding:6px 12px;border-radius:4px;margin-bottom:12px;text-align:center;font-size:0.8rem}
.st.ok{background:rgba(0,255,136,0.1);color:#00ff88}
.st.err{background:rgba(255,87,87,0.1);color:#ff5757}
.empty{text-align:center;color:#666;padding:40px;font-size:0.9rem}
@media(max-width:1000px){.grid{grid-template-columns:1fr}.charts{grid-template-columns:1fr}}
</style></head><body>
<div class="container">
<h1>Pile Stiffness Degradation &mdash; Target vs Predicted</h1>
<div id="st" class="st ok">Loading...</div>
<div class="grid">
<div class="card">
<h2>Test Scenarios</h2>
<ul class="sc-list" id="scList"></ul>
</div>
<div class="card">
<div id="detail">
<div class="empty">Select a scenario from the list</div>
</div>
</div>
</div>
</div>
<script>
let scenarios=[], charts={};
const fmt=n=>Math.abs(n)>=1e9?(n/1e9).toFixed(2)+'e9':Math.abs(n)>=1e6?(n/1e6).toFixed(2)+'e6':Math.abs(n)>=1e3?(n/1e3).toFixed(2)+'e3':n.toFixed(2);

function mkChart(canvasId, label, targetData, predData, steps){
    if(charts[canvasId]){charts[canvasId].destroy()}
    const ctx=document.getElementById(canvasId);
    charts[canvasId]=new Chart(ctx,{type:'line',data:{labels:steps,datasets:[
        {label:'Target',data:targetData,borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,0.05)',fill:false,tension:0.3,pointRadius:2,pointBackgroundColor:'#00ff88',borderWidth:2},
        {label:'Predicted',data:predData,borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,0.05)',fill:false,tension:0.3,pointRadius:2,pointBackgroundColor:'#ff6b6b',borderWidth:2,borderDash:[5,3]}
    ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,callbacks:{label:function(c){return c.dataset.label+': '+fmt(c.parsed.y)}}}},scales:{x:{title:{display:true,text:'Step',color:'#aaa'},grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#aaa',maxTicksLimit:15}},y:{title:{display:true,text:label,color:'#aaa'},grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#aaa',callback:v=>fmt(v)}}},interaction:{mode:'nearest',axis:'x',intersect:false}}});
}

function showScenario(idx){
    document.querySelectorAll('.sc-list li').forEach((li,i)=>{li.classList.toggle('active',i===idx)});
    const s=scenarios[idx];
    const paramNames={'PI':'PI','Gmax':'Gmax (Pa)','v':'v','Dp':'Dp (m)','Tp':'Tp (m)','Lp':'Lp (m)','Ip':'Ip','Dp_Lp':'Dp/Lp'};
    let paramsHtml='<div class="params-bar">';
    for(const[k,v]of Object.entries(s.params)){paramsHtml+=`<div class="param-chip"><span class="lbl">${paramNames[k]||k}:</span> <span class="val">${fmt(v)}</span></div>`}
    paramsHtml+='</div>';
    document.getElementById('detail').innerHTML=paramsHtml+
        '<div class="legend-bar"><div class="legend-item"><div class="legend-dot" style="background:#00ff88"></div>Target (Real)</div><div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div>Predicted</div></div>'+
        '<div class="charts"><div class="chc"><div class="cht">KL &mdash; Lateral Stiffness</div><canvas id="cKL"></canvas></div>'+
        '<div class="chc"><div class="cht">KR &mdash; Rotational Stiffness</div><canvas id="cKR"></canvas></div>'+
        '<div class="chc"><div class="cht">KLR &mdash; Cross-Coupling</div><canvas id="cKLR"></canvas></div></div>';
    mkChart('cKL','KL',s.target.KL,s.predicted.KL,s.steps);
    mkChart('cKR','KR',s.target.KR,s.predicted.KR,s.steps);
    mkChart('cKLR','KLR',s.target.KLR,s.predicted.KLR,s.steps);
}

window.onload=async()=>{
    const st=document.getElementById('st');
    try{
        const r=await(await fetch('/api/scenarios')).json();
        if(r.error){st.textContent=r.error;st.className='st err';return}
        scenarios=r.scenarios;
        st.textContent=`Model loaded — ${scenarios.length} test scenarios`;
        const list=document.getElementById('scList');
        scenarios.forEach((s,i)=>{
            const li=document.createElement('li');
            li.innerHTML=`<div>${s.label}</div><div class="sc-params">PI=${fmt(s.params.PI)} Gmax=${fmt(s.params.Gmax)} Dp=${fmt(s.params.Dp)}</div>`;
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
    return jsonify({'scenarios': scenarios_cache})


if __name__ == '__main__':
    print("=" * 40)
    if load_all():
        print(f"Model loaded | {len(scenarios_cache)} test scenarios | Seq: {max_seq_len}")
    else:
        print("Run: python train.py first")
    print("Server: http://127.0.0.1:5000")
    print("=" * 40)
    app.run(host='127.0.0.1', port=5000, debug=True)
