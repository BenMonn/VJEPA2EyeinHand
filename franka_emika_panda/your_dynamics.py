# your_dynamics.py
# Tiny latent dynamics model Φ: (z_t, a_t) -> z_{t+1}
# Save/load helpers and an optional trainer for supervised regression on logged data.

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyDynamicsMLP(nn.Module):
    def __init__(self, z_dim: int = 512, a_dim: int = 7, hidden: int = 1024, depth: int = 2):
        super().__init__()
        dims = [z_dim + a_dim] + [hidden] * (depth - 1) + [z_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.GELU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, D)
        a_t: (B, A)
        returns z_{t+1} prediction: (B, D), L2-normalized for stability
        """
        x = torch.cat([z_t, a_t], dim=-1)
        z_tp1 = self.net(x)
        z_tp1 = F.normalize(z_tp1, dim=-1, eps=1e-8)
        return z_tp1

@dataclass
class DynConfig:
    z_dim: int = 512
    a_dim: int = 7
    hidden: int = 1024
    depth: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-6


class LatentDynamics(nn.Module):
    """
    Wrapper that also stores action bounds (optional) and provides a simple train_step.
    """
    def __init__(self, cfg: DynConfig):
        super().__init__()
        self.cfg = cfg
        self.phi = TinyDynamicsMLP(cfg.z_dim, cfg.a_dim, cfg.hidden, cfg.depth)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        return self.phi(z_t, a_t)

    def loss(self, z_pred: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        # MSE in normalized latent space
        z_tp1_n = F.normalize(z_tp1, dim=-1, eps=1e-8)
        return F.mse_loss(z_pred, z_tp1)

    def train_step(self, batch, opt: torch.optim.Optimizer, *, debug: bool = False, step: Optional[int] = None):
        """
        batch: dict with 'z_t','a_t','z_tp1' tensors on the same device
        """
        z_t   = batch["z_t"]
        a_t   = batch["a_t"]
        z_tp1 = batch["z_tp1"]

        if debug and (step is None or step % 200 == 0):
            with torch.no_grad():
                print(f"[dyn] z_t std={z_std().item():.6f} z_tp1 std={z_tp1.std().item():.6f} a std={a_t.std().item():.6f}")

        z_pred = self.forward(z_t, a_t)
        loss = self.loss(z_pred, z_tp1)

        if not torch.isfinite(loss):
            if debug:
                with torch.no_grad():
                    sim = F.cosine_similarity(z_pred, F.normalize(z_tp1, dim=-1, eps=1e-6), dim=-1)
                    print("[dyn] non-finite loss; stats:",
                          f"z_pred.std={z_pred.std().item():.6f}",
                          f"z_tp1.std={z_tp1.std().item():.6f}",
                          f"mean cos={sim.mean().item():.6f}")
            return float("nan")

        opt.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        opt.step()
        return float(loss.item())


# ---------------------------
# Save / Load helpers
# ---------------------------

def save_Phi(dyn: LatentDynamics, path: str):
    torch.save({
        "cfg": dyn.cfg.__dict__,
        "state_dict": dyn.state_dict()
    }, path)

def load_Phi(path: str, device: str = "cpu") -> LatentDynamics:
    blob = torch.load(path, map_location=device)
    cfg = DynConfig(**blob["cfg"])
    dyn = LatentDynamics(cfg)
    dyn.load_state_dict(blob["state_dict"])
    dyn.to(device).eval()
    return dyn


# ---------------------------
# Minimal training loop utility (optional)
# ---------------------------

def fit_latent_dynamics(
    model: LatentDynamics,
    loader,                         # iterable of dicts {'z_t','a_t','z_tp1'}
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: str = "cuda"
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model.train()
    step = 0
    for ep in range(1, epochs + 1):
        running = 0.0
        n = 0
        for batch in loader:
            # Ensure tensors are on device & normalized
            b = {k: v.to(device) for k, v in batch.items()}
            loss = model.train_step(b, opt)
            step += 1
            if not (loss == loss):
                continue
            running += loss
            n += 1
        print(f"[Dyn] Epoch {ep}/{epochs}  loss={running/max(n,1):.6f}")
    model.eval()
