import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sym_posdef(h: torch.Tensor) -> torch.Tensor:
    h = 0.5 * (h + h.transpose(-1, -2))
    h = h + 0.1 * torch.eye(3, device=h.device)
    return h


def _sym_nonneg(h: torch.Tensor) -> torch.Tensor:
    h = 0.5 * (h + h.transpose(-1, -2))
    h = torch.clamp(h, min=0.0)
    return h


class SlotAttention(nn.Module):
    def __init__(self, n_slots: int, dim: int, iters: int = 3):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.iters = iters

        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, n_slots, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        self.last_attn = None

    def forward(self, inputs: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        n, t, d = inputs.shape
        inputs = self.norm_inputs(inputs)

        mu = self.slots_mu.expand(n, -1, -1)
        sigma = torch.exp(self.slots_logsigma).expand(n, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        k = self.to_k(inputs)
        v = self.to_v(inputs)

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))

            attn_logits = torch.einsum("nid,njd->nij", q, k) / math.sqrt(d)
            attn = F.softmax(attn_logits, dim=-1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            updates = torch.einsum("nij,njd->nid", attn, v)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d),
            ).reshape(n, self.n_slots, d)

            slots = slots + self.mlp(self.norm_mlp(slots))

        self.last_attn = attn
        if return_attn:
            return slots, attn
        return slots, None


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=True)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff))
        return x, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, self_attn = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=True)
        x = self.norm1(x + self.dropout(attn_out))

        attn_out, cross_attn = self.cross_attn(x, memory, memory, need_weights=True)
        x = self.norm2(x + self.dropout(attn_out))

        ff = self.linear2(F.relu(self.linear1(x)))
        x = self.norm3(x + self.dropout(ff))
        return x, self_attn, cross_attn


class SelfCalibrationTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_yield: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_yield = n_yield
        self.n_slots = n_yield + 1
        self.d_model = d_model
        self.n_heads = n_heads

        self.static_embed = nn.Linear(10, d_model)
        self.load_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.stiffness_embed = nn.Linear(9, d_model)

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, dropout) for _ in range(n_enc_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, dropout) for _ in range(n_dec_layers)]
        )

        self.slot_attn = SlotAttention(self.n_slots, d_model, iters=3)

        self.h0_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 9))
        self.hn_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 9))
        self.gate_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 2))

        self.mass_head = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
        self.f0_in = nn.Linear(1, d_model)
        self.f0_out = nn.Linear(d_model, 1)

    def forward(
        self,
        x_static: torch.Tensor,
        load_seq: torch.Tensor,
        f0_prev: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, dict, Optional[Dict[str, List[torch.Tensor]]]]:
        n, t, _ = load_seq.shape

        static_tok = self.static_embed(x_static).unsqueeze(1).expand(-1, t, -1)
        load_tok = self.load_embed(load_seq)
        tokens = static_tok + load_tok + self.pos_embed

        slots, attn = self.slot_attn(tokens, return_attn=return_attn)

        h0_vec = self.h0_head(slots[:, 0, :])
        h0 = h0_vec.view(n, 3, 3)
        h0 = _sym_posdef(h0)

        hn_vec = self.hn_head(slots[:, 1:, :])
        hn = hn_vec.view(n, self.n_yield, 3, 3)
        hn = _sym_nonneg(hn)

        gate_params = self.gate_head(slots[:, 1:, :])
        a = gate_params[:, :, 0:1] * 4.0
        b = gate_params[:, :, 1:2]

        load_t = load_seq.transpose(1, 2)
        w = torch.sigmoid(a * load_t + b)

        h0_exp = h0.unsqueeze(1)
        hn_exp = hn.unsqueeze(2)
        w_exp = w.unsqueeze(-1).unsqueeze(-1)
        hij = h0_exp.unsqueeze(2) - torch.sum(w_exp * hn_exp, dim=1)
        hij = torch.clamp(hij, min=1e-3)

        hij_flat = hij.reshape(n, t, 9)
        stiffness_tok = self.stiffness_embed(hij_flat)
        enc_in = tokens + stiffness_tok

        enc_attn = []
        for layer in self.enc_layers:
            enc_in, attn_w = layer(enc_in)
            enc_attn.append(attn_w)

        if f0_prev is None:
            f0_prev = torch.zeros(n, t, 1, device=load_seq.device)
        dec_in = self.f0_in(f0_prev)
        tgt_mask = _causal_mask(t, load_seq.device)

        dec_attn = []
        cross_attn = []
        dec_out = dec_in
        for layer in self.dec_layers:
            dec_out, self_w, cross_w = layer(dec_out, enc_in, tgt_mask)
            dec_attn.append(self_w)
            cross_attn.append(cross_w)

        f0 = self.f0_out(dec_out).squeeze(-1)

        m_eff = torch.exp(self.mass_head(x_static)) + 1.0
        k_eff = hij.diagonal(dim1=-2, dim2=-1).sum(-1)
        f0 = (1.0 / (2.0 * math.pi)) * torch.sqrt(k_eff / m_eff)

        aux = {"h0": h0, "hn": hn, "w": w, "hij": hij, "m_eff": m_eff}
        attn_map = None
        if return_attn:
            attn_map = {
                "slot_to_token": attn,
                "enc_self": enc_attn,
                "dec_self": dec_attn,
                "dec_cross": cross_attn,
            }
        return f0, aux, attn_map


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask
