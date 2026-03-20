# Thermodynamic 21-Slot Model

**Building a load-driven soil degradation prediction system with thermodynamic constraints.**

Based on hand-sketch physics principles from your notes:
- $S_1 = \{G_0, H, e\}$ (initial state)
- $S_2 - S_1 = 4H_i[j]_{n=1}$ (state change from load history)

## Architecture

### 21 Specialized Slots
- **GROUP 1 (Slots 0-6): Load Analysis** — Horizontal force (H), Moment (M), cyclic loading, cumulative strain
- **GROUP 2 (Slots 7-13): Soil Degradation** — Gmax, stiffness, strain, energy dissipation, entropy, strength, void ratio
- **GROUP 3 (Slots 14-19): Pile/Structural** — Geometry, bending stiffness (EI), soil-pile interaction, displacement, stiffness drop, environment
- **GROUP 4 (Slot 20): Fusion** — Cross-attention over all 20 specialized slots (3 refinement iterations)

### Output Targets (Primary: **G/Gmax**)
1. **G/Gmax** — Soil shear modulus ratio [0.1, 1.0] ← **WEIGHTED 2.0** (primary)
2. **C_ratio** — Lateral stiffness ratio [0.05, 1.0]
3. **E_dissipated** — Energy loss [0, 1]
4. **entropy** — Entropy change (2nd law constraint)

## Physics Principles

### Hardin-Drnevich Degradation (1972)
$$\frac{G}{G_0} = \frac{1}{1 + (\gamma/\gamma_{ref})^{0.9}}$$

where $\gamma$ = cyclic shear strain accumulated from load history.

### Thermodynamic Constraints
- **1st Law:** Energy dissipation increases with load
  $$E_{dissipated} = 1 - \frac{G}{G_0} \geq 0$$

- **2nd Law:** Entropy increases (irreversible deformation)
  $$S_{change} = -\ln(G/G_0) \geq 0$$

### Stiffness Drop
Lateral stiffness drops exponentially as load increases:
$$C(t) = C_{max} \cdot \frac{G(t)}{G_0} \cdot e^{-0.5\gamma(t)}$$

## Files

- **`thermodynamic_21slot_model.py`** — Main model (generates data, defines architecture, trains)
- **`architecture.tex`** — Detailed LaTeX documentation of the 21-slot system
- **`data/`** — Generated CSV files (after running):
  - `pile.csv` — Pile properties
  - `soil.csv` — Soil properties (G0, Su, void ratio, OCR, degradation factor)
  - `load.csv` — Load evolution (H, M, cycles, cumulative strain)
  - `environment.csv` — Environmental conditions
  - `ground_truth.csv` — Target outputs (G/Gmax, C_ratio, E_dissipated, entropy)

## Usage

```bash
cd c:\Users\youss\Downloads\PFE\meeting 2\thermodynamic_slot_model
python thermodynamic_21slot_model.py
```

### What it does:
1. Generates 200 scenarios × 30 time steps of synthetic data
2. Data embeds physics: stiffness drops as load increases, G/Gmax degrades
3. Trains the 21-slot model for 250 epochs
4. Evaluates on held-out test set
5. Prints metrics for all 4 outputs

## Key Differences from 5-Slot Model

| Aspect | 5-Slot | 21-Slot |
|--------|--------|---------|
| Specialized slots | 4 (CSV domains) | 20 (finer granularity) |
| Load representation | One slot | 7 slots (low/mid/high H, M, cyclic, cumulative H/M) |
| Soil degradation | One slot | 7 slots (Gmax, stiffness, strain, energy, entropy, Su, void) |
| Pile response | Implicit | 6 slots (geometry, EI, interaction, displacement, stiffness drop, env) |
| **Primary output focus** | 4 equal targets | **G/Gmax weighted 2.0** |
| Physics constraints | Smooth activation | Smooth + thermodynamic penalties |

## Next Steps

You can extend this to:
- Add more load types (torsion, combined H+M interactions)
- Implement fully differentiable thermodynamic loss (dE ≥ 0, dS ≥ dQ/T)
- Visualize attention maps showing which load level triggers which slot
- Compare predictions vs. physical laboratory cyclic test data
