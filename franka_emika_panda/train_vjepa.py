# train_vjepa.py (core)
import torch

for step, clip in enumerate(loader):  # clip: [B, T, 3, H, W]
    # Sample context/future views, masks
    ctx_view, fut_view, fut_mask = sample_views_and_masks(clip)  # masking in space-time

    # Encode target latent (stop-grad)
    with torch.no_grad():
        z_tgt = h_xi(fut_view)                 # [B, T', D]
        z_tgt_masked = apply_mask(z_tgt, fut_mask)

    # Context encoding + prediction
    z_ctx = f_theta(ctx_view)                  # [B, T_ctx, D]
    z_pred = g_theta(z_ctx, fut_mask)          # predict masked future tokens

    loss = mse(z_pred, z_tgt_masked)
    loss.backward(); opt.step(); opt.zero_grad()

    # EMA update
    for p_tgt, p_ctx in zip(h_xi.parameters(), f_theta.parameters()):
        p_tgt.data.mul_(tau).add_((1-tau)*p_ctx.data)
