# your_vjepa.py
# Minimal V-JEPA components: tiny ViT encoder (shared arch), EMA target, predictor MLP,
# and helper functions for preprocessing, encoding, save/load.
#
# Requirements: torch, torchvision (only for transforms if you prefer; not strictly required)
#
# Notes:
# - Input: RGB PIL Image or np.uint8 array (H,W,3), expected 224x224. We'll resize if needed.
# - Latent dim D defaults to 512.
# - The predictor here is a simple MLP that operates on a pooled latent (global). For a stronger
#   model you can switch to token-wise prediction or add time masking—this keeps it runnable.

import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Image preprocessing
# ---------------------------

IM_SIZE = 224
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _to_tensor(img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """
    Convert PIL or np.uint8(H,W,3) to torch.FloatTensor (1,3,H,W) in [0,1].
    """
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            # assume already float [0,1] or [0,255]
            arr = img
            if arr.max() > 1.0:
                arr = np.clip(arr, 0, 255) / 255.0
            arr = arr.astype(np.float32)
        else:
            arr = img.astype(np.float32) / 255.0
        if arr.ndim == 3:
            H, W, C = arr.shape
            assert C == 3, "Expected 3 channels"
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        else:
            raise ValueError("np.ndarray must be HxWx3")
        return tensor
    elif isinstance(img, Image.Image):
        img = img.convert("RGB")
        tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)  # (H,W,3)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        return tensor
    else:
        raise TypeError("img must be PIL.Image or np.ndarray")

def _resize_if_needed(x: torch.Tensor, size: int = IM_SIZE) -> torch.Tensor:
    # x: (B,3,H,W)
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

def preprocess_np_or_pil(img, device: str = "cpu") -> torch.Tensor:
    """
    Returns normalized tensor (1,3,224,224) on the requested device.
    """
    x = _to_tensor(img)                        # (1,3,H,W) on CPU
    x = _resize_if_needed(x, IM_SIZE)          # still CPU
    x = x.to(device, non_blocking=True)        # 👉 move first
    mean = IM_MEAN.to(device, non_blocking=True)
    std  = IM_STD.to(device, non_blocking=True)
    return (x - mean) / std

# ---------------------------
# Tiny ViT-style encoder
# ---------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=384, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: (B,3,H,W)
        x = self.proj(x)    # (B, D, H/ps, W/ps)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x, (H, W)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        # x: (B,N,D)
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + x2
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=6, num_heads=6, patch=16, out_dim=512, drop=0.0):
        super().__init__()
        self.patch = patch
        self.embed = PatchEmbed(in_ch=3, embed_dim=embed_dim, patch=patch)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=4.0, drop=drop, attn_drop=0.0)
        for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)  # global projection

        # learnable CLS token (optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # simple: no absolute pos; rely on conv patch + attn

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B,3,224,224)
        tokens, (H, W) = self.embed(x)     # (B, N, D)
        B, N, D = tokens.shape

        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, tokens], dim=1)     # (B,1+N,D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]               # (B, D)
        z = self.head(cls_out)          # (B, out_dim)
        z = F.normalize(z, dim=-1, eps=1e-6)      # helpful for stability
        return z                         # global latent


# ---------------------------
# Predictor (maps context latent -> predicted future latent)
# ---------------------------

class PredictorMLP(nn.Module):
    def __init__(self, dim=512, hidden=1024, depth=2, drop=0.0):
        super().__init__()
        layers = []
        d = dim
        for i in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop)]
            d = hidden
        layers += [nn.Linear(d, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_ctx):  # (B,D) -> (B,D)
        out = self.net(z_ctx)
        out = F.normalize(out, dim=-1, eps=1e-6)
        return out


# ---------------------------
# V-JEPA container
# ---------------------------

@dataclass
class VJEPAConfig:
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    patch: int = 16
    out_dim: int = 512
    predictor_hidden: int = 1024
    predictor_depth: int = 2
    ema_tau: float = 0.996

class VJEPA(nn.Module):
    def __init__(self, cfg: VJEPAConfig):
        super().__init__()
        self.cfg = cfg
        # context / online encoder f_theta
        self.ctx_enc = ViTEncoder(
            embed_dim=cfg.embed_dim, depth=cfg.depth,
            num_heads=cfg.num_heads, patch=cfg.patch,
            out_dim=cfg.out_dim
        )
        # target encoder h_xi (EMA copy)
        self.tgt_enc = ViTEncoder(
            embed_dim=cfg.embed_dim, depth=cfg.depth,
            num_heads=cfg.num_heads, patch=cfg.patch,
            out_dim=cfg.out_dim
        )
        # initialize target = context
        self._copy_params(self.tgt_enc, self.ctx_enc)

        # predictor g_theta
        self.predictor = PredictorMLP(
            dim=cfg.out_dim, hidden=cfg.predictor_hidden, depth=cfg.predictor_depth
        )

        # EMA tau
        self.register_buffer("ema_tau", torch.tensor(cfg.ema_tau, dtype=torch.float32))

        # stop-grad convenience
        for p in self.tgt_enc.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _copy_params(self, target: nn.Module, source: nn.Module):
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.data.copy_(p_s.data)

    @torch.no_grad()
    def ema_update(self):
        tau = float(self.ema_tau.item())
        for p_t, p_s in zip(self.tgt_enc.parameters(), self.ctx_enc.parameters()):
            p_t.data.mul_(tau).add_(p_s.data, alpha=(1.0 - tau))

    def forward(self, ctx_img: torch.Tensor, fut_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training-time call:
          - ctx_img, fut_img: (B,3,224,224)
          - returns (pred_latent, target_latent) with target stop-grad
        """
        z_ctx = self.ctx_enc(ctx_img)          # (B,D)
        z_pred = self.predictor(z_ctx)         # (B,D)

        with torch.no_grad():
            z_tgt = self.tgt_enc(fut_img)      # (B,D), stop-grad

        return z_pred, z_tgt


# ---------------------------
# Factory & helpers
# ---------------------------

def build_vjepa_small(out_dim: int = 512) -> VJEPA:
    cfg = VJEPAConfig(out_dim=out_dim)
    return VJEPA(cfg)

@torch.no_grad()
def encode_image(tgt_encoder: nn.Module, img: Union[Image.Image, np.ndarray], device: str = "cpu") -> torch.Tensor:
    """
    Returns (1,D) latent using the *target* encoder.
    """
    x = preprocess_np_or_pil(img, device=device)
    z = tgt_encoder(x)
    return z  # (1,D)

@torch.no_grad()
def encode_clip(tgt_encoder: nn.Module, frames: np.ndarray, device: str = "cpu", reduce: str = "mean") -> torch.Tensor:
    """
    frames: np.uint8 array (T,H,W,3) or float [0..1]; returns (1,D) by pooling over time.
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3, "frames must be (T,H,W,3)"
    T = frames.shape[0]
    zs = []
    for t in range(T):
        z = encode_image(tgt_encoder, frames[t], device=device)  # (1,D)
        zs.append(z)
    Z = torch.cat(zs, dim=0)  # (T,D)
    if reduce == "mean":
        z = Z.mean(dim=0, keepdim=True)
    elif reduce == "last":
        z = Z[-1:, :]
    else:
        raise ValueError("reduce must be 'mean' or 'last'")
    z = F.normalize(z, dim=-1, eps=1e-6)
    return z  # (1,D)


# ---------------------------
# Save / Load utilities
# ---------------------------

def save_target_encoder(vjepa: VJEPA, path: str):
    state = {"tgt_enc": vjepa.tgt_enc.state_dict()}
    torch.save(state, path)

def load_target_encoder(path: str, device: str = "cpu") -> nn.Module:
    model = build_vjepa_small().tgt_enc
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["tgt_enc"])
    model.to(device).eval()
    return model

def save_full(vjepa: VJEPA, path: str):
    torch.save({"cfg": vjepa.cfg.__dict__,
                "ctx_enc": vjepa.ctx_enc.state_dict(),
                "tgt_enc": vjepa.tgt_enc.state_dict(),
                "predictor": vjepa.predictor.state_dict()}, path)

def load_full(path: str, device: str = "cpu") -> VJEPA:
    blob = torch.load(path, map_location=device)
    cfg = VJEPAConfig(**blob["cfg"])
    vjepa = VJEPA(cfg)
    vjepa.ctx_enc.load_state_dict(blob["ctx_enc"])
    vjepa.tgt_enc.load_state_dict(blob["tgt_enc"])
    vjepa.predictor.load_state_dict(blob["predictor"])
    vjepa.to(device)
    return vjepa
