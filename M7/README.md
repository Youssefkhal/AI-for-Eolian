# M7: Ψ-NN Applied to Pile Stiffness Degradation

Implements the 3-stage pipeline from **Liu et al. (Nature Communications, 2025)**:
_"Automatic network structure discovery of physics informed neural networks via knowledge distillation"_

## Pipeline

### Stage A — Physics-Informed Distillation
- **Teacher**: M6 `SlotAttentionDegradation` (21 slots, 56,646 params, R²≈0.98)
- **Student**: Same architecture, trained with:
  - `L_distill` = MSE(student output, teacher output)
  - `L_data` = MSE(student output, ground truth)
  - `L_reg` = L1 norm on slot vectors (encourages sparsity)
- **Purpose**: Separate physics learning (teacher) from regularization (student)

### Stage B — Structure Extraction
- Extract refined slot vectors from trained student
- Compute pairwise **cosine similarity** between 20 drop-slots
- **K-means clustering** with elbow method + silhouette scores → find k* prototypes
- Build **relation matrix R** [20 × k*] describing how each slot maps to prototypes
- Discover: which slots are redundant, reused, or scaled versions of others

### Stage C — Structured Ψ-Model
- Only **k\* + 1** trainable slot vectors (1 initial + k* prototypes)
- All 20 drops reconstructed via `R @ prototypes * scale_factors`
- Retrained with distillation + data + physics constraints
- Result: fewer parameters, same physics, structured knowledge

## Usage

```bash
# Train the full pipeline (Stage A → B → C)
python train.py

# Launch the web dashboard
python webapp.py
# Open http://127.0.0.1:5000
```

## Output Files

| File | Description |
|------|-------------|
| `pile_model.pth` | Ψ-model weights |
| `psi_config.pkl` | k*, relation matrix R, centroids |
| `psi_discovery.json` | Full analysis: similarity matrix, clustering, slot norms |
| `comparison.json` | M6 Teacher vs Student vs Ψ-model metrics |
| `model_metrics.pkl` | Ψ-model per-slot per-variable metrics |
| `test_data.pkl` | Test set for webapp |

## Architecture

```
M6 Teacher (frozen)
    ↓ distill
Stage-A Student (L1 regularized)
    ↓ analyze slots
Stage-B: K-means on 20 drop-slots → k* prototypes + relation matrix R
    ↓ reconstruct
Stage-C: Ψ-Model
    1 initial slot + k* prototype slots
    20 drops = R @ prototypes × scale
    → cross-attn → self-attn → MLP (×3 iterations)
    → initial_proj(slot 1) = K⁰
    → drop_proj(slots 2-21) = ΔK
    → physics: ΔKL,ΔKR ≤ 0, ΔKLR ≥ 0
    → cumsum → output [B, 21, 3]
```

## Webapp Dashboard

The webapp shows:
- **Ψ-NN Discovery panel**: k*, silhouette scores, prototype clusters, cosine similarity heatmap
- **Model comparison table**: M6 Teacher vs Student vs Ψ-Model (params, R², RMSE, MAE)
- **Per-scenario charts**: KL, KR, KLR trajectories (target vs predicted)
- **Per-slot accuracy table**: R², RMSE, MAE per slot and per variable
