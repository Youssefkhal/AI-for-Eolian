"""
M10: XAI Engine — Attention Rollout, LRP, Token Attribution Maps
================================================================
Implements three complementary XAI methods for the SwiGLU Ψ-NN:

  1. Attention Rollout — aggregates cross-attn and self-attn separately
       Cross: traces input→slot information flow across iterations
       Self:  traces slot→slot information flow with residual mixing

  2. LRP (Layer-wise Relevance Propagation) — gradient-based attribution
       Propagates all the way back to the 8 raw input features.
       Uses Gradient × Input (satisfies first-order completeness axiom).

  3. Integrated Gradients — path-based attribution (gold standard)
       Satisfies the completeness axiom exactly:
       f(x) - f(baseline) = Σⱼ attributionⱼ

All three methods produce per-variable (KL/KR/KLR) feature attribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────
# Attention Extraction
# ─────────────────────────────────────────────────────

def forward_with_attention(model, x, seq_len=None):
    """
    Run the Ψ-model forward pass while capturing all attention weight matrices.

    Args:
        model: SlotAttentionPsiModel instance
        x: input tensor [B, 8]
        seq_len: sequence length (default: model.max_seq_len)

    Returns:
        predictions:  [B, 21, 3]  output stiffness curves
        cross_attns:  list of num_iterations tensors, each [B, 21, 1]
        self_attns:   list of num_iterations tensors, each [B, 21, 21]
        slots_history: list of num_iterations tensors, each [B, 21, 64]
    """
    if seq_len is None:
        seq_len = model.max_seq_len
    B = x.size(0)
    x_embed = model.input_embed(x).unsqueeze(1)  # [B, 1, 64]

    # Build initial slot bank
    if hasattr(model, 'initial_slot'):
        # Ψ-model with prototypes
        initial = model.initial_slot.expand(B, -1, -1)
        drops = model.reconstruct_drop_slots(B)
        slots = torch.cat([initial, drops], dim=1)  # [B, 21, 64]
    else:
        # Student or teacher model
        slots = model.slots.expand(B, -1, -1)

    cross_attns = []
    self_attns = []
    slots_history = []

    for _ in range(model.num_iterations):
        # Cross-attention: slots attend to input embedding
        cross_out, cross_w = model.cross_attn(
            slots, x_embed, x_embed,
            need_weights=True, average_attn_weights=True
        )
        slots = model.cross_norm(slots + cross_out)
        cross_attns.append(cross_w.detach())  # [B, 21, 1]

        # Self-attention: slots attend to each other
        self_out, self_w = model.self_attn(
            slots, slots, slots,
            need_weights=True, average_attn_weights=True
        )
        slots = model.self_norm(slots + self_out)
        self_attns.append(self_w.detach())  # [B, 21, 21]

        # SwiGLU MLP + residual
        slots = model.mlp_norm(slots + model.slot_mlp(slots))
        slots_history.append(slots.detach().clone())

    # Prediction heads (same logic as model.forward)
    init_pred = model.initial_proj(slots[:, 0:1, :])
    raw_drops = model.drop_proj(slots[:, 1:, :])
    drops_kl_kr = -torch.abs(raw_drops[:, :, :2])
    drops_klr = torch.abs(raw_drops[:, :, 2:3])
    constrained = torch.cat([drops_kl_kr, drops_klr], dim=2)
    predictions = torch.cat([
        init_pred,
        init_pred + torch.cumsum(constrained, dim=1)
    ], dim=1)

    return predictions, cross_attns, self_attns, slots_history


# ─────────────────────────────────────────────────────
# Attention Rollout (Cross + Self, Separately)
# ─────────────────────────────────────────────────────

def compute_cross_attention_rollout(cross_attns):
    """
    Rollout across cross-attention iterations.

    Since there is only 1 source token (the input embedding), each
    cross-attention map [B, 21, 1] tells us how much each slot "pulls"
    from the input. We average across iterations and normalize.

    Returns:
        rollout:   [B, 21]  per-slot importance from cross-attention
        per_iter:  list of [B, 21] per-iteration cross-attention
    """
    per_iter = []
    for attn in cross_attns:
        per_iter.append(attn.squeeze(-1).cpu())  # [B, 21]

    # Aggregate: average across iterations
    stacked = torch.stack(per_iter, dim=0)   # [num_iter, B, 21]
    rollout = stacked.mean(dim=0)            # [B, 21]

    # Normalize to sum to 1 across slots
    rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return rollout, per_iter


def compute_self_attention_rollout(self_attns, residual_weight=0.5):
    """
    Rollout across self-attention iterations with residual mixing.

    For each iteration's self-attention A_l (shape [B, 21, 21]):
        A'_l  = (1 - w) · I  +  w · A_l
    Then multiply across iterations:
        R_total = A'_L  ×  A'_{L-1}  × ··· ×  A'_1

    Args:
        self_attns:      list of [B, 21, 21] attention matrices
        residual_weight: mixing weight (0 = all residual, 1 = all attention)

    Returns:
        rollout:   [B, 21, 21]  aggregated slot-to-slot flow
        per_iter:  list of [B, 21, 21]  per-iteration attention
    """
    B = self_attns[0].size(0)
    N = self_attns[0].size(1)  # 21
    identity = torch.eye(N).unsqueeze(0).expand(B, -1, -1)

    per_iter = [a.cpu() for a in self_attns]

    rollout = None
    for attn in per_iter:
        mixed = (1 - residual_weight) * identity + residual_weight * attn
        # Re-normalize rows to sum to 1
        mixed = mixed / mixed.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if rollout is None:
            rollout = mixed
        else:
            rollout = torch.bmm(mixed, rollout)

    return rollout, per_iter


# ─────────────────────────────────────────────────────
# LRP: Gradient × Input  (to 8 raw features)
# ─────────────────────────────────────────────────────

def compute_gradient_x_input(model, x, num_steps=21):
    """
    Compute Gradient × Input attribution (LRP-equivalent).

    For each output variable (KL/KR/KLR) and degradation step:
        attribution_j  =  x_j  ×  ∂output / ∂x_j

    This satisfies the first-order completeness (conservation) axiom:
        f(x) ≈ f(0) + Σ_j  x_j · ∂f/∂x_j

    Args:
        model: SlotAttentionPsiModel (eval mode)
        x: [1, 8] input tensor (scaled features)
        num_steps: output time steps (21)

    Returns:
        attribution: numpy array [3, 21, 8]
                     (variable × step × feature relevance)
    """
    model.eval()
    n_feat = x.shape[1]
    attribution = np.zeros((3, num_steps, n_feat))

    for vi in range(3):
        for si in range(num_steps):
            x_input = x.clone().detach().requires_grad_(True)
            output = model(x_input, seq_len=num_steps)
            target = output[0, si, vi]

            model.zero_grad()
            target.backward()

            grad = x_input.grad.detach().squeeze()   # [8]
            attr = (grad * x.detach().squeeze()).cpu().numpy()
            attribution[vi, si] = attr

    return attribution


# ─────────────────────────────────────────────────────
# Integrated Gradients  (gold standard LRP)
# ─────────────────────────────────────────────────────

def compute_integrated_gradients(model, x, num_steps=21, ig_steps=50):
    """
    Integrated Gradients — path-based attribution that exactly satisfies
    the completeness axiom:

        f(x) - f(baseline) = Σ_j  attribution_j

    where attribution_j = (x_j - baseline_j) ×
                          ∫₀¹ ∂f/∂x_j (baseline + α(x - baseline)) dα

    Args:
        model: SlotAttentionPsiModel
        x: [1, 8] input tensor
        num_steps: output time steps (21)
        ig_steps:  integration steps (higher = more accurate)

    Returns:
        attribution: numpy array [3, 21, 8]
    """
    model.eval()
    baseline = torch.zeros_like(x)
    n_feat = x.shape[1]
    attribution = np.zeros((3, num_steps, n_feat))

    for vi in range(3):
        for si in range(num_steps):
            total_grad = torch.zeros_like(x)

            for step in range(ig_steps):
                alpha = (step + 0.5) / ig_steps
                interp = baseline + alpha * (x - baseline)
                interp = interp.clone().detach().requires_grad_(True)

                output = model(interp, seq_len=num_steps)
                target = output[0, si, vi]

                model.zero_grad()
                target.backward()
                total_grad += interp.grad.detach()

            avg_grad = total_grad / ig_steps
            attr = ((x - baseline) * avg_grad).squeeze().cpu().numpy()
            attribution[vi, si] = attr

    return attribution


# ─────────────────────────────────────────────────────
# XAI Analyzer  (orchestrator)
# ─────────────────────────────────────────────────────

class XAIAnalyzer:
    """
    Orchestrates all XAI methods for scenario-level explainability.

    Usage:
        analyzer = XAIAnalyzer(model, feature_names)
        result   = analyzer.analyze(x_scaled_input)
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = (
            list(feature_names) if not isinstance(feature_names, list)
            else feature_names
        )

    def analyze(self, x_input, method='gradient'):
        """
        Run complete XAI analysis for one input scenario.

        Args:
            x_input: numpy array or tensor [8] or [1, 8]
            method:  'gradient' (fast, ~50 ms) or
                     'integrated_gradients' (accurate, ~2 s)

        Returns:
            dict ready for JSON serialization containing:
              - cross_attention_rollout    [21]
              - cross_attention_per_iter   [3][21]
              - self_attention_rollout     [21][21]
              - self_attention_per_iter    [3][21][21]
              - lrp_attribution            {KL:[21][8], KR:..., KLR:...}
              - feature_importance         {KL:[8], KR:[8], KLR:[8], combined:[8]}
              - feature_names              [8]
        """
        x = (torch.FloatTensor(x_input)
             if not isinstance(x_input, torch.Tensor) else x_input)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # ── 1. Forward with attention extraction ──
        with torch.no_grad():
            preds, cross_attns, self_attns, slots_hist = \
                forward_with_attention(self.model, x)

        # ── 2. Attention rollout (cross and self, separately) ──
        cross_rollout, cross_per_iter = \
            compute_cross_attention_rollout(cross_attns)
        self_rollout, self_per_iter = \
            compute_self_attention_rollout(self_attns)

        # ── 3. LRP attribution back to 8 raw features ──
        if method == 'integrated_gradients':
            lrp_attr = compute_integrated_gradients(
                self.model, x, ig_steps=30)
        else:
            lrp_attr = compute_gradient_x_input(self.model, x)

        # ── 4. Aggregate feature importance per variable ──
        var_names = ['KL', 'KR', 'KLR']
        feature_importance = {}
        for vi, vn in enumerate(var_names):
            abs_attr = np.abs(lrp_attr[vi])          # [21, 8]
            total = abs_attr.sum(axis=0)              # [8]
            s = total.sum()
            pct = (total / s * 100) if s > 0 else total
            feature_importance[vn] = pct.tolist()

        # Combined importance (average across variables)
        combined = np.mean(
            [np.abs(lrp_attr[vi]).sum(axis=0) for vi in range(3)],
            axis=0,
        )
        s = combined.sum()
        combined_pct = (combined / s * 100) if s > 0 else combined
        feature_importance['combined'] = combined_pct.tolist()

        # ── 5. Build JSON-serialisable response ──
        return {
            'cross_attention_rollout': cross_rollout[0].numpy().tolist(),
            'cross_attention_per_iter': [
                ci[0].numpy().tolist() for ci in cross_per_iter
            ],
            'self_attention_rollout': self_rollout[0].numpy().tolist(),
            'self_attention_per_iter': [
                si[0].numpy().tolist() for si in self_per_iter
            ],
            'lrp_attribution': {
                'KL':  lrp_attr[0].tolist(),
                'KR':  lrp_attr[1].tolist(),
                'KLR': lrp_attr[2].tolist(),
            },
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
        }
