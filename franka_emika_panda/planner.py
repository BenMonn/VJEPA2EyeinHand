# planner.py
import numpy as np
import torch

@torch.no_grad()
def cem_plan(
    z_t: torch.Tensor,            # (1, D) latent at current step
    z_goal: torch.Tensor,         # (1, D) latent goal
    Phi,                          # LatentDynamics (forward: Phi(z, a) -> z')
    act_low: np.ndarray,          # (A,)
    act_high: np.ndarray,         # (A,)
    H: int = 12,
    K: int = 512,
    elites: int = 64,
    iters: int = 3,
    action_l2: float = 0.0,       # small action magnitude penalty
    device: str = None,
):
    """
    Cross-Entropy Method in latent space.
    Returns the *first* action of the best sequence as a numpy array (A,).
    """
    assert z_t.ndim == 2 and z_goal.ndim == 2, "z_t and z_goal must be (1,D)"
    D = z_t.shape[1]
    A = act_low.shape[0]
    if device is None:
        device = z_t.device

    # bounds as tensors
    lo = torch.as_tensor(act_low, dtype=torch.float32, device=device)  # (A,)
    hi = torch.as_tensor(act_high, dtype=torch.float32, device=device) # (A,)

    # Initialize Gaussian over action sequences: mean=0, std = (hi-lo)/4
    mean = torch.zeros(H, A, device=device)
    std  = (hi - lo).unsqueeze(0).repeat(H, 1) / 4.0  # (H,A)

    best_seq = None
    best_cost = float("inf")

    for _ in range(iters):
        # Sample K sequences: (K,H,A)
        eps = torch.randn(K, H, A, device=device)
        acts = mean.unsqueeze(0) + std.unsqueeze(0) * eps  # (K,H,A)
        # Clip to bounds
        acts = torch.max(torch.min(acts, hi), lo)

        # Rollout all K sequences and accumulate intermediate cost
        z = z_t.expand(K, D)  # (K,D)
        total_cost = torch.zeros(K, device=device)

        for t in range(H):
            a_t = acts[:, t, :]              # (K,A)
            z = Phi(z, a_t)                  # (K,D)
            # compute running cost at each step (encourage progress)
            goal = z_goal.expand(K, D)
            step_cost = torch.sum((z - goal) ** 2, dim=1)  # (K,)
            total_cost += step_cost

        # Average over horizon length (optional smoothing)
        cost = total_cost / H

        # Add small penalty for big actions
        if action_l2 > 0.0:
            cost = cost + action_l2 * torch.sum(acts**2, dim=(1, 2))

        # Select elites
        topk = torch.topk(-cost, k=elites, dim=0).indices  # negate to get smallest
        elite_acts = acts[topk, :, :]  # (elites,H,A)

        # Update distribution
        mean = elite_acts.mean(dim=0)  # (H,A)
        std  = elite_acts.std(dim=0) + 1e-6  # (H,A)

        # Track best
        cur_best = cost.min().item()
        if cur_best < best_cost:
            best_cost = cur_best
            best_seq = elite_acts[0]  # any elite is fine; we'll use mean at the end

    # Use mean of final distribution (more stable) for action
    a0 = mean[0].clamp(min=lo, max=hi)  # (A,)
    return a0.detach().cpu().numpy()

