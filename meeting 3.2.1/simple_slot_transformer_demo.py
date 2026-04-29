import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        n, t, d = inputs.shape
        inputs = self.norm_inputs(inputs)

        mu = self.slots_mu.expand(n, -1, -1)
        sigma = torch.exp(self.slots_logsigma).expand(n, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)

        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            attn_logits = torch.einsum("nid,njd->nij", q, k) / math.sqrt(d)
            attn = F.softmax(attn_logits, dim=-1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.einsum("nij,njd->nid", attn, v)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(n, self.n_slots, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class SimpleSlotTransformer(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 64, n_yield: int = 20):
        super().__init__()
        self.seq_len = seq_len
        self.n_yield = n_yield
        self.n_slots = n_yield + 1

        self.load_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.slot_attn = SlotAttention(self.n_slots, d_model, iters=3)
        self.h0_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 9))
        self.hn_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 9))
        self.gate_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 2))

        enc_layer = nn.TransformerEncoderLayer(d_model, 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.f0_head = nn.Linear(d_model, 1)

    def forward(self, load_seq: torch.Tensor) -> torch.Tensor:
        n, t, _ = load_seq.shape
        tokens = self.load_embed(load_seq) + self.pos_embed

        slots = self.slot_attn(tokens)
        h0 = self.h0_head(slots[:, 0, :]).view(n, 3, 3)
        hn = self.hn_head(slots[:, 1:, :]).view(n, self.n_yield, 3, 3)
        gates = self.gate_head(slots[:, 1:, :])
        a = gates[:, :, 0:1] * 4.0
        b = gates[:, :, 1:2]

        w = torch.sigmoid(a * load_seq.transpose(1, 2) + b)
        hij = h0.unsqueeze(1) - torch.sum(w.unsqueeze(-1).unsqueeze(-1) * hn.unsqueeze(2), dim=1)
        hij = torch.clamp(hij, min=1e-3)

        k_eff = hij.diagonal(dim1=-2, dim2=-1).sum(-1)
        f0_proxy = (1.0 / (2.0 * math.pi)) * torch.sqrt(k_eff)

        enc_out = self.encoder(tokens)
        f0 = self.f0_head(enc_out).squeeze(-1)

        return f0, f0_proxy


def demo():
    torch.manual_seed(7)
    model = SimpleSlotTransformer(seq_len=50)
    load_seq = torch.rand(8, 50, 1)

    pred_f0, phys_f0 = model(load_seq)
    loss = F.mse_loss(pred_f0, phys_f0)
    loss.backward()

    print("ok", pred_f0.shape, phys_f0.shape)


if __name__ == "__main__":
    demo()
