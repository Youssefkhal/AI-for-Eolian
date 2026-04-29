import os
import sys
import threading
import time
from typing import Any, Dict

from flask import Flask, jsonify, render_template_string, request

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from config import TrainConfig
from train import train

app = Flask(__name__)

STATE: Dict[str, Any] = {
    "running": False,
    "last_error": None,
    "last_run": None,
}


def _serialize_tensor(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def _load_results() -> Dict[str, Any]:
    outputs_dir = os.path.join(ROOT_DIR, "outputs")
    metrics_path = os.path.join(outputs_dir, "metrics.pt")
    attn_path = os.path.join(outputs_dir, "attention_maps.pt")
    rollout_path = os.path.join(outputs_dir, "attention_rollout.pt")

    if not (os.path.exists(metrics_path) and os.path.exists(attn_path) and os.path.exists(rollout_path)):
        return {"ok": False, "error": "Outputs not found. Run training first."}

    metrics = torch.load(metrics_path, map_location="cpu")
    attn = torch.load(attn_path, map_location="cpu")
    rollout = torch.load(rollout_path, map_location="cpu")

    slot_to_token = attn.get("slot_to_token")
    if slot_to_token is None:
        return {"ok": False, "error": "Missing attention maps."}

    heatmap = _serialize_tensor(slot_to_token[0])
    rollout_vec = _serialize_tensor(rollout[0])

    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    response = {
        "ok": True,
        "metrics": {
            "nrmse": _to_float(metrics["full_metrics"]["nrmse"]),
            "mape": _to_float(metrics["full_metrics"]["mape"]),
            "r2": _to_float(metrics["full_metrics"]["r2"]),
            "cov": _to_float(metrics["cov"]),
        },
        "heatmap": heatmap,
        "rollout": rollout_vec,
    }
    return response


def _run_training(fast: bool) -> None:
    try:
        STATE["running"] = True
        STATE["last_error"] = None
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        if fast:
            os.environ["FAST_RUN"] = "1"
        else:
            os.environ.pop("FAST_RUN", None)

        train(TrainConfig())
        STATE["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        STATE["last_error"] = str(exc)
    finally:
        STATE["running"] = False


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.get("/")
def index():
        return render_template_string(
                """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Post-typhoon OWT Digital Twin</title>
        <style>
            :root {
                --bg: #0c0f14;
                --surface: #121826;
                --text: #f9fafb;
                --text-muted: #9ca3af;
                --accent: #10b981;
                --accent-2: #3b82f6;
                --shadow: rgba(15, 23, 42, 0.4);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: "Segoe UI", sans-serif;
                color: var(--text);
                background: radial-gradient(circle at top, #1f2937 0%, #0c0f14 55%);
                min-height: 100vh;
                line-height: 1.6;
            }
            header {
                padding: 3rem 8vw 2rem;
                display: grid;
                gap: 1.5rem;
            }
            h1 { font-size: clamp(2rem, 3vw, 3rem); }
            .lead { color: var(--text-muted); max-width: 48rem; }
            .btn {
                padding: 0.7rem 1.4rem;
                border-radius: 999px;
                background: linear-gradient(120deg, var(--accent), #22c55e);
                color: #0b1f17;
                border: none;
                font-weight: 600;
                cursor: pointer;
            }
            .section { padding: 3rem 8vw; }
            .grid { display: grid; gap: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
            .card {
                background: var(--surface);
                padding: 1.4rem;
                border-radius: 1rem;
                box-shadow: 0 20px 40px var(--shadow);
            }
            .pill {
                display: inline-block;
                padding: 0.3rem 0.7rem;
                margin: 0.3rem 0.2rem 0 0;
                border-radius: 999px;
                background: rgba(16, 185, 129, 0.15);
                color: var(--accent);
                font-size: 0.75rem;
            }
            .status {
                margin-top: 1rem;
                padding: 0.6rem 0.8rem;
                border-radius: 0.6rem;
                background: rgba(16, 185, 129, 0.15);
                color: var(--accent);
                font-size: 0.9rem;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .label { display: block; font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }
            canvas {
                width: 100%;
                height: 220px;
                border-radius: 1rem;
                background: rgba(255, 255, 255, 0.04);
            }
            code { display: inline-block; margin-top: 0.6rem; padding: 0.4rem 0.6rem; border-radius: 0.5rem; background: rgba(0,0,0,0.35); }
        </style>
    </head>
    <body>
        <header>
            <h1>Post-typhoon OWT Digital Twin</h1>
            <p class="lead">
                Physics-informed slot-attention + encoder-decoder transformer to model
                natural frequency shifts with attention-based explainability.
            </p>
            <button id="run-backend" class="btn">Run Backend (Fast)</button>
            <p id="backend-status" class="status">Idle</p>
        </header>

        <section class="section">
            <div class="grid">
                <div class="card">
                    <h3>Architecture</h3>
                    <p>Slot-attention learns $H_0$ and $H_n$ with gates $w_n(t)$, then the transformer forecasts $f_0^{OWT}(t)$.</p>
                    <div class="pill">TIM blueprint</div>
                    <div class="pill">Masked decoder</div>
                    <div class="pill">Cross-attention</div>
                </div>
                <div class="card">
                    <h3>Physics Losses</h3>
                    <p>LoT1: frequency consistency. LoT2: monotone, non-negative stiffness degradation.</p>
                    <div class="pill">LoT1</div>
                    <div class="pill">LoT2</div>
                    <div class="pill">Positivity</div>
                </div>
                <div class="card">
                    <h3>Code Map</h3>
                    <p>Data: meeting 3.2/data.py</p>
                    <p>Model: meeting 3.2/model.py</p>
                    <p>Training: meeting 3.2/train.py</p>
                    <p>Backend: meeting 3.2/backend/app.py</p>
                </div>
            </div>
        </section>

        <section class="section">
            <div class="grid">
                <div class="card">
                    <h3>Metrics</h3>
                    <div class="metrics">
                        <div><span class="label">NRMSE</span><span id="metric-nrmse">0.00</span></div>
                        <div><span class="label">MAPE</span><span id="metric-mape">0.00</span></div>
                        <div><span class="label">R2</span><span id="metric-r2">0.00</span></div>
                        <div><span class="label">CoV</span><span id="metric-cov">0.00</span></div>
                    </div>
                </div>
                <div class="card">
                    <h3>Slot Attention Heatmap</h3>
                    <canvas id="heatmap"></canvas>
                </div>
                <div class="card">
                    <h3>Attention Roll-out</h3>
                    <canvas id="rollout"></canvas>
                </div>
            </div>
        </section>

        <section class="section">
            <div class="card">
                <h3>API Endpoints</h3>
                <p><code>POST /api/run?fast=1</code></p>
                <p><code>GET /api/status</code></p>
                <p><code>GET /api/results</code></p>
            </div>
        </section>

        <script>
            const metricEls = {
                nrmse: document.getElementById("metric-nrmse"),
                mape: document.getElementById("metric-mape"),
                r2: document.getElementById("metric-r2"),
                cov: document.getElementById("metric-cov"),
            };
            const heatmapCanvas = document.getElementById("heatmap");
            const rolloutCanvas = document.getElementById("rollout");
            const statusEl = document.getElementById("backend-status");
            const runBtn = document.getElementById("run-backend");

            function drawHeatmap(canvas, matrix) {
                const ctx = canvas.getContext("2d");
                const rows = matrix.length;
                const cols = matrix[0].length;
                canvas.width = cols;
                canvas.height = rows;
                const imageData = ctx.createImageData(cols, rows);
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        const value = Math.min(1, Math.max(0, matrix[r][c]));
                        const idx = (r * cols + c) * 4;
                        imageData.data[idx] = 40 + value * 180;
                        imageData.data[idx + 1] = 100 + value * 120;
                        imageData.data[idx + 2] = 255 - value * 120;
                        imageData.data[idx + 3] = 255;
                    }
                }
                ctx.putImageData(imageData, 0, 0);
            }

            function drawRollout(canvas, series) {
                const ctx = canvas.getContext("2d");
                const width = canvas.clientWidth;
                const height = canvas.clientHeight;
                canvas.width = width * 2;
                canvas.height = height * 2;
                ctx.scale(2, 2);
                ctx.clearRect(0, 0, width, height);
                ctx.strokeStyle = "#10b981";
                ctx.lineWidth = 2;
                ctx.beginPath();
                series.forEach((v, i) => {
                    const x = (i / (series.length - 1)) * width;
                    const y = height - v * height;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            function updateUI(payload) {
                metricEls.nrmse.textContent = payload.metrics.nrmse.toFixed(3);
                metricEls.mape.textContent = payload.metrics.mape.toFixed(3);
                metricEls.r2.textContent = payload.metrics.r2.toFixed(2);
                metricEls.cov.textContent = payload.metrics.cov.toFixed(2);
                drawHeatmap(heatmapCanvas, payload.heatmap);
                drawRollout(rolloutCanvas, payload.rollout);
            }

            async function fetchJson(url, options) {
                const res = await fetch(url, options);
                return await res.json();
            }

            async function pollStatus() {
                try {
                    const status = await fetchJson("/api/status");
                    if (status.running) {
                        statusEl.textContent = "Running...";
                        setTimeout(pollStatus, 1500);
                        return;
                    }
                    if (status.last_error) {
                        statusEl.textContent = `Error: ${status.last_error}`;
                        return;
                    }
                    statusEl.textContent = status.last_run ? `Done: ${status.last_run}` : "Idle";
                    const results = await fetchJson("/api/results");
                    if (results.ok) {
                        updateUI(results);
                    }
                } catch (err) {
                    statusEl.textContent = "Backend not reachable";
                }
            }

            runBtn.addEventListener("click", async () => {
                statusEl.textContent = "Starting...";
                await fetchJson("/api/run?fast=1", { method: "POST" });
                pollStatus();
            });
        </script>
    </body>
</html>
                """
        )


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "running": STATE["running"], "last_error": STATE["last_error"]})


@app.get("/api/status")
def status():
    return jsonify({"running": STATE["running"], "last_error": STATE["last_error"], "last_run": STATE["last_run"]})


@app.post("/api/run")
def run():
    if STATE["running"]:
        return jsonify({"ok": False, "error": "Training already running."}), 409

    fast = request.args.get("fast", "1") == "1"
    thread = threading.Thread(target=_run_training, args=(fast,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "started": True, "fast": fast})


@app.get("/api/results")
def results():
    return jsonify(_load_results())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
