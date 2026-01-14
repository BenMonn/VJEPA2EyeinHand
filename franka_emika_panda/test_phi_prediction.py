# test_phi_prediction.py
import torch, numpy as np
from your_vjepa import load_target_encoder, preprocess_np_or_pil
from your_dynamics import load_Phi

# === Paths ===
ENCODER_CKPT = "checkpoints/vjepa_target_ep03.pth"
PHI_CKPT = "checkpoints/phi_latent.pth"
TEST_EP = "dataset/ep_00000.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load models ===
encoder = load_target_encoder(ENCODER_CKPT, device=DEVICE)
phi = load_Phi(PHI_CKPT, device=DEVICE)
encoder.eval(); phi.eval()

# === Load one episode ===
ep = np.load(TEST_EP)
frames = ep["frames"]
acts = ep["actions"]
T = min(len(frames)-1, len(acts)-1)

# === Encode all frames ===
print(f"Encoding {T+1} frames ...")
zs = []
with torch.no_grad():
    for t in range(T+1):
        x = preprocess_np_or_pil(frames[t], device=DEVICE)
        z = encoder(x)   # shape (1,D)
        zs.append(z)
zs = torch.cat(zs, dim=0)  # (T+1, D)
A = torch.tensor(acts[:T], dtype=torch.float32, device=DEVICE)  # (T, a_dim)

# === Predict next latents ===
with torch.no_grad():
    z_t = zs[:-1]
    z_tp1_pred = phi(z_t, A)
    z_tp1_true = zs[1:]
    mse = torch.mean((z_tp1_pred - z_tp1_true)**2).item()

print(f"Mean squared prediction error: {mse:.12f}")

# ---- 5-step rollout test (k=5) ----
k = 5
T_k = T - (k - 1)
z_pred_k = []
with torch.no_grad():
    z_cur = zs[0:T_k]          # (T-k+1, D)
    for i in range(k):
        a_cur = torch.tensor(acts[i:i+T_k], dtype=torch.float32, device=DEVICE)  # align actions
        z_cur = phi(z_cur, a_cur)   # one-step forward
    z_pred_k = z_cur                # predicted z_{t+k}
    z_true_k = zs[k: k+T_k]         # ground-truth z_{t+k}

mse_k = torch.mean((z_pred_k.double() - z_true_k.double())**2).item()
cos_k = torch.nn.functional.cosine_similarity(z_pred_k, z_true_k, dim=-1).mean().item()

print(f"k-step (k={k}) MSE: {mse_k:.6e}")
print(f"k-step (k={k}) cosine: {cos_k:.6f}")

# Optional: visualize similarity
sim = torch.nn.functional.cosine_similarity(z_tp1_pred, z_tp1_true, dim=-1)
print(f"Cosine similarity (first 10): {sim[:10].cpu().numpy()}")
print(f"Average cosine similarity: {sim.mean().item():.4f}")

# --- Zero-MSE diagnostics ---
print("\n--- Zero-MSE diagnostics ---")
with torch.no_grad():
    z_t    = zs[:-1]
    z_tp1t = zs[1:]
    z_pred = phi(z_t, A)

    print("z_t.std:", z_t.std().item())
    print("z_tp1.std:", z_tp1t.std().item())
    print("z_pred.std:", z_pred.std().item())
    print("equal(pred,true)?", torch.equal(z_pred, z_tp1t))

    cos = torch.nn.functional.cosine_similarity(
        z_pred,
        torch.nn.functional.normalize(z_tp1t, dim=-1, eps=1e-6),
        dim=-1
    )
    print("mean cosine similarity:", cos.mean().item())
