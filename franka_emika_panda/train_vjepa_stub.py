# train_vjepa_stub.py
# Tiny training loop for V-JEPA using random crops/jitter + optional cutout mask.
# Relies only on torch, numpy, PIL, and the your_vjepa.py we wrote earlier.

import os, random, argparse, time
from glob import glob

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from your_vjepa import (
    build_vjepa_small,
    preprocess_np_or_pil,  # (1,3,224,224) normalized
    save_target_encoder,
)

x = preprocess_np_or_pil(np.zeros((224, 224, 3), dtype=np.uint8), device="cuda")
print(x.isfinite().all())

# ---------------------------
# Simple augmentations (no torchvision)
# ---------------------------

def random_resized_crop(img: Image.Image, min_scale=0.6, max_scale=1.0, out_size=224):
    w, h = img.size
    scale = random.uniform(min_scale, max_scale)
    tw, th = int(w * scale), int(h * scale)
    if tw < 1: tw = 1
    if th < 1: th = 1
    if tw < w:
        x0 = random.randint(0, w - tw)
    else:
        x0 = 0
    if th < h:
        y0 = random.randint(0, h - th)
    else:
        y0 = 0
    img = img.crop((x0, y0, x0 + tw, y0 + th))
    return img.resize((out_size, out_size), Image.BILINEAR)

def color_jitter(img: Image.Image, b=0.2, c=0.2, s=0.2, h=0.0):
    # brightness/contrast/saturation; (hue skipped to keep it simple)
    if b > 0:
        img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-b, b))
    if c > 0:
        img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-c, c))
    if s > 0:
        img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-s, s))
    return img

def random_horizontal_flip(img: Image.Image, p=0.5):
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img

def random_cutout(img: Image.Image, p=0.5, min_frac=0.1, max_frac=0.3, fill=(123, 123, 123)):
    if random.random() >= p:
        return img
    w, h = img.size
    fw = int(w * random.uniform(min_frac, max_frac))
    fh = int(h * random.uniform(min_frac, max_frac))
    x0 = random.randint(0, max(0, w - fw))
    y0 = random.randint(0, max(0, h - fh))
    patch = Image.new("RGB", (fw, fh), fill)
    img = img.copy()
    img.paste(patch, (x0, y0))
    return img

def to_pil(frame_np: np.ndarray) -> Image.Image:
    # frame_np: (H, W, 3), uint8 or float in [0,1]
    if frame_np.dtype != np.uint8:
        arr = np.clip(frame_np, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        arr = frame_np
    return Image.fromarray(arr, mode="RGB")


# ---------------------------
# Dataset
# ---------------------------

class ClipPairs(Dataset):
    """
    Loads .npz clips with shape [T, H, W, 3].
    Returns two *different time* frames: ctx_frame, fut_frame (PIL),
    each with separate augmentations; fut_frame may get cutout (mask).
    """
    def __init__(self, root_dir, ctx_min_stride=1, ctx_max_stride=4, fut_min_dt=3, fut_max_dt=12):
        self.paths = sorted(glob(os.path.join(root_dir, "*.npz")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .npz clips found under {root_dir}")
        self.ctx_min_stride = ctx_min_stride
        self.ctx_max_stride = ctx_max_stride
        self.fut_min_dt = fut_min_dt
        self.fut_max_dt = fut_max_dt

    def __len__(self):
        return len(self.paths)

    def _pick_times(self, T):
        # pick t_ctx, then t_fut > t_ctx by [fut_min_dt..fut_max_dt]
        if T < self.fut_min_dt + 2:
            # very short clips: fallback to neighboring frames
            t_ctx = max(0, T // 3)
            t_fut = min(T - 1, t_ctx + 1)
            return t_ctx, t_fut
        t_ctx = random.randint(0, max(0, T - self.fut_min_dt - 1))
        dt = random.randint(self.fut_min_dt, min(self.fut_max_dt, T - 1 - t_ctx))
        t_fut = t_ctx + dt
        return t_ctx, t_fut

    def __getitem__(self, idx):
        clip = np.load(self.paths[idx])  # [T,H,W,3]
        assert clip.ndim == 4 and clip.shape[-1] == 3, "Clip must be (T,H,W,3)"
        T = clip.shape[0]
        t_ctx, t_fut = self._pick_times(T)

        ctx = to_pil(clip[t_ctx])
        fut = to_pil(clip[t_fut])

        # separate augs
        ctx = random_resized_crop(ctx, 0.6, 1.0, 224)
        ctx = random_horizontal_flip(ctx, 0.5)
        ctx = color_jitter(ctx, 0.2, 0.2, 0.2)

        fut = random_resized_crop(fut, 0.6, 1.0, 224)
        fut = random_horizontal_flip(fut, 0.5)
        fut = color_jitter(fut, 0.2, 0.2, 0.2)
        fut = random_cutout(fut, p=0.5, min_frac=0.15, max_frac=0.3)  # "mask"

        # Defer normalization/resize to preprocess_np_or_pil (keeps it consistent)
        return ctx, fut


# ---------------------------
# Training
# ---------------------------

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, opt, device="cuda", ema_every=1, print_every=50):
    model.train()
    running = 0.0
    n = 0
    loss_fn = nn.MSELoss()

    for step, (ctx_img_pil, fut_img_pil) in enumerate(loader, 1):
        # preprocess to (1,3,224,224) per sample, then batch
        ctx_batch = torch.cat([preprocess_np_or_pil(x, device=device) for x in ctx_img_pil], dim=0)
        fut_batch = torch.cat([preprocess_np_or_pil(x, device=device) for x in fut_img_pil], dim=0)

        z_pred, z_tgt = model(ctx_batch, fut_batch)  # (B,D), (B,D)
        loss = loss_fn(z_pred, z_tgt)

        if not torch.isfinite(loss):
            print("  ⚠️  skipping batch with non-finite loss:", float(loss.item()))
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        # EMA on target encoder
        if (step % ema_every) == 0:
            model.ema_update()

        running += float(loss.item())
        n += 1

        if (step % print_every) == 0:
            print(f"  step {step:05d} | loss {running/n:.6f}")

    return running / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Folder with *.npz clips [T,H,W,3]")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    # Dataset / Loader
    ds = ClipPairs(args.data)
    # Collate returns list of PILs -> we’ll preprocess inside the loop (keeps transforms simple)
    def _collate(batch):
        ctxs, futs = zip(*batch)
        return list(ctxs), list(futs)

    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=_collate
    )

    # Model & Optim
    model = build_vjepa_small().to(args.device)
    opt = torch.optim.AdamW([
        {"params": model.ctx_enc.parameters(), "lr": args.lr},
        {"params": model.predictor.parameters(), "lr": args.lr},
        # target encoder is EMA (frozen)
    ], weight_decay=args.wd)

    print(f"Starting training for {args.epochs} epochs on {args.device} "
          f"with {len(ds)} clips, batch={args.batch}")

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, opt, device=args.device, print_every=50)
        print(f"[epoch {ep}/{args.epochs}] avg_loss={avg_loss:.6f}")

        # Save the EMA target encoder each epoch (what you'll use at control time)
        tgt_path = os.path.join(args.save_dir, f"vjepa_target_ep{ep:02d}.pth")
        save_target_encoder(model, tgt_path)
        print(f"  saved target encoder → {tgt_path}")

    print(f"Done. Total time: {(time.time()-t0)/60.0:.1f} min")

if __name__ == "__main__":
    main()
