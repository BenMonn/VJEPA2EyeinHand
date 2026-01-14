# fit_phi_from_dataset.py
import os, argparse, torch, numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader

from your_vjepa import load_target_encoder, preprocess_np_or_pil
from your_dynamics import LatentDynamics, DynConfig, fit_latent_dynamics, save_Phi

class LatentTriples(Dataset):
    """Dataset that loads .npz clips with (frames, actions) and builds (z_t, a_t, z_tp1) tuples."""
    def __init__(self, root, encoder, device="cuda"):
        self.paths = sorted(glob(os.path.join(root, "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files found in {root}")
        self.enc = encoder
        self.device = device

        self.z_t, self.a_t, self.z_tp1 = [], [], []
        with torch.no_grad():
            for p in self.paths:
                blob = np.load(p)
                frames = blob["frames"]       # (T, H, W, 3)
                acts   = blob["actions"]      # (T, A)
                T = min(len(frames) - 1, len(acts) - 1)
                if T <= 0:
                    continue
                # Encode all frames
                z_all = []
                for t in range(T + 1):
                    x = preprocess_np_or_pil(frames[t], device=device)  # (1,3,224,224)
                    z = self.enc(x).detach().cpu()
                    z_all.append(z)
                Z = torch.cat(z_all, dim=0)   # (T+1, D)
                A = torch.from_numpy(acts[:T]).float()  # (T, A)
                self.z_t.append(Z[:-1])
                self.z_tp1.append(Z[1:])
                self.a_t.append(A)
        # Concatenate all episodes
        self.z_t   = torch.cat(self.z_t, dim=0)
        self.z_tp1 = torch.cat(self.z_tp1, dim=0)
        self.a_t   = torch.cat(self.a_t, dim=0)
        print(f"Loaded {len(self.z_t)} latent pairs from {len(self.paths)} episodes")

    def __len__(self):
        return self.z_t.shape[0]

    def __getitem__(self, i):
        return {"z_t": self.z_t[i], "a_t": self.a_t[i], "z_tp1": self.z_tp1[i]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset folder with .npz files")
    ap.add_argument("--encoder_ckpt", required=True, help="Path to V-JEPA target encoder .pth")
    ap.add_argument("--out", default="checkpoints/phi_latent.pth")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load pretrained V-JEPA encoder (frozen)
    h_xi = load_target_encoder(args.encoder_ckpt, device=args.device)
    h_xi.eval()

    # Dataset + DataLoader
    ds = LatentTriples(args.data, h_xi, device=args.device)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)

    # Latent dynamics Φ
    dyn = LatentDynamics(DynConfig(z_dim=512, a_dim=ds.a_t.shape[1]))
    fit_latent_dynamics(dyn, loader, epochs=args.epochs, device=args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_Phi(dyn, args.out)
    print("Saved latent dynamics model →", args.out)

if __name__ == "__main__":
    main()
