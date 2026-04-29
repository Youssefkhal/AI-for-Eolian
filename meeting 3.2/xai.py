from typing import Dict, List

import torch


def compute_attention_maps(model, x_static: torch.Tensor, load_seq: torch.Tensor, f0_prev: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        _, _, attn_map = model(x_static, load_seq, f0_prev=f0_prev, return_attn=True)

    return {
        "slot_to_token": attn_map["slot_to_token"],
        "enc_self": torch.stack(attn_map["enc_self"], dim=0),
        "dec_self": torch.stack(attn_map["dec_self"], dim=0),
        "dec_cross": torch.stack(attn_map["dec_cross"], dim=0),
    }


def attention_rollout(attn_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
    # enc_self: (L, N, T, T), dec_cross: (L, N, T, T)
    enc_self = attn_maps["enc_self"].mean(dim=0)
    enc_roll = enc_self.mean(dim=0)
    enc_roll = enc_roll / (enc_roll.sum(dim=-1, keepdim=True) + 1e-8)

    dec_cross = attn_maps["dec_cross"].mean(dim=0)
    dec_cross_mean = dec_cross.mean(dim=1)
    dec_cross_mean = dec_cross_mean / (dec_cross_mean.sum(dim=-1, keepdim=True) + 1e-8)

    rollout = torch.matmul(dec_cross_mean, enc_roll)
    rollout = rollout.mean(dim=1)
    rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-8)
    return rollout


def save_heatmap(slot_to_token: torch.Tensor, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    heatmap = slot_to_token[0].detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(label="attention")
    plt.xlabel("time")
    plt.ylabel("slot")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_rollout(rollout: torch.Tensor, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    r = rollout[0].detach().cpu().numpy()
    plt.figure(figsize=(8, 3))
    plt.plot(r)
    plt.xlabel("time")
    plt.ylabel("importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
