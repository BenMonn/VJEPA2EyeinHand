# run_global_vjepa_control.py
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import torch
import mujoco
from PIL import Image

from camera_setup import make_joint_observation, get_cam_id
from planner import cem_plan
from your_vjepa import load_target_encoder, encode_image, preprocess_np_or_pil
from your_dynamics import load_Phi

W = 224
H = 224

# Default action bounds for 7 Panda joints (joint-vel control)
ACT_LOW   = np.array([-0.4]*7, dtype=np.float32)
ACT_HIGH  = np.array([ 0.4]*7, dtype=np.float32)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # --- Load model & camera ---
    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # (optional) you can fetch cam ids if you still want to jitter them explicitly
    cam_global = get_cam_id(model, "global")
    cam_eye    = get_cam_id(model, "eye_in_hand")


    print("Initial qpos:", data.qpos[:7].copy())
    for i in range(200):
        data.ctrl[:7] = np.array([0.5, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mujoco.mj_step(model, data)
    print("Post-reset qpos:", data.qpos[:7].copy())

    for _ in range(50):
        data.qvel[:7] = np.array([0.3, -0.2, 0.15, 0, 0, 0, 0], dtype=np.float32)
        mujoco.mj_step(model, data)

    def small_runtime_randomization(model):
        # jitter both global and eye_in_hand if present
        if hasattr(model, "cam_pos"):
            for cam_name in ("global", "eye_in_hand"):
                try:
                    cid = get_cam_id(model, cam_name)
                except ValueError:
                    continue
                model.cam_pos[cid, :] += np.random.uniform(-0.005, 0.005, size=3)

        if getattr(model, "nlight", 0) > 0 and hasattr(model, "light_pos"):
            for lid in range(model.nlight):
                model.light_pos[lid, :] += np.random.uniform(-0.02, 0.02, size=3)


    small_runtime_randomization(model)

    # --- Load networks ---
    h_xi = load_target_encoder(args.encoder_ckpt, device=device).eval()
    Phi  = load_Phi(args.phi_ckpt, device=device).eval()

    # --- Build goal latent ---
    goal_img = Image.open(args.goal_img).convert("RGB").resize((224, 224))
    z_goal = encode_image(h_xi, goal_img, device=device).detach()

    def settle_hold_current_target(steps=60):
        mujoco.mj_forward(model, data)
        for _ in range(steps):
            data.ctrl[:7] = data.qpos[:7].astype(np.float32)
            mujoco.mj_step(model, data)

    def try_sampled_offsets_away_from_goal(threshold=0.97, base_mag=0.20, tries=8, samples_per_try=16):
        q0 = data.qpos[:7].copy()
        best_cos, best_offset = 1.0, None

        for t in range(tries):
            mag = base_mag + 0.05 * t
            for _ in range(samples_per_try):
                v = np.random.randn(7).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-9)
                offset = v * mag

                data.qpos[:7] = q0 + offset
                settle_hold_current_target(steps=40)

                # img = render_rgb(model, data, cam_id, width=224, height=224)
                joint_img = make_joint_observation(model, data, width=W, height=H)
                # joint_img.shape = (H, 2W, 3)

                # typical PyTorch transform (example)
                obs = joint_img.astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))  # (C, H, W) = (3, H, 2W)
                obs = torch.from_numpy(obs)

                z = h_xi(preprocess_np_or_pil(joint_img, device=device)).detach()
                cos = torch.nn.functional.cosine_similarity(z, z_goal).item()

                if cos < best_cos:
                    best_cos, best_offset = cos, offset
                    if best_cos < threshold:
                        data.qpos[:7] = q0 + best_offset
                        settle_hold_current_target(steps=60)
                        return True, best_cos

            data.qpos[:7] = q0
            settle_hold_current_target(steps=20)

        if best_offset is not None:
            data.qpos[:7] = q0 + best_offset
            settle_hold_current_target(steps=60)
            return False, best_cos

        return False, 1.0
    
    # --- Ensure start is sufficiently different from goal ---
    ok, cos0 = try_sampled_offsets_away_from_goal(
        threshold=0.90, 
        base_mag=0.40, 
        tries=8, 
        samples_per_try=16
    )
    print(("Start accepted" if ok else "Start suboptimal") + f" (cos={cos0:.3f})")

    # (optional) small camera jitter so image always differs slightly
    if hasattr(model, "cam_pos"):
        for cam_name in ("global", "eye_in_hand"):
            try:
                cid = get_cam_id(model, cam_name)
            except ValueError:
                continue  # camera doesn't exist in model
            model.cam_pos[cid, :] += np.random.uniform(-0.01, 0.01, size=3)


    # --- Control loop (CEM/MPC in latent space) ---
    print("Starting CEM control ...")

    cost_log = []
    frames_out = []

    # --- Snapshot before control starts (step 0) ---
    os.makedirs("snapshots", exist_ok=True)
    # img0 = render_rgb(model, data, cam_id, width=224, height=224)
    joint_img0 = make_joint_observation(model, data, width=W, height=H)
    # joint_img.shape = (H, 2W, 3)

    # typical PyTorch transform (example)
    obs = joint_img0.astype(np.float32) / 255.0
    obs = np.transpose(obs, (2, 0, 1))  # (C, H, W) = (3, H, 2W)
    obs = torch.from_numpy(obs)

    Image.fromarray(joint_img0).save("snapshots/step000.png")
    print("Saved snapshot → snapshots/step000.png")

    motion_log = []
    qpos_prev = data.qpos.copy()

    # --- Position-target control setup ---
    DT = model.opt.timestep
    FRAME_SKIP = 10
    POS_GAIN = 20.0                    # try 5–15
    VEL_GAIN = 0.5                     # try 0.1–1.0
    ACTION_SCALE = 10.0

    JOINT_MIN = np.array([-2.8, -2.8, -2.8, -2.8, -2.8, -2.8, -2.8], dtype=np.float32)
    JOINT_MAX = np.array([ 2.8,  2.8,  2.8,  2.8,  2.8,  2.8,  2.8], dtype=np.float32)

    qpos_target = data.qpos[:7].copy()  # initialize target
    qpos_target = np.clip(qpos_target, JOINT_MIN, JOINT_MAX) # joint limits

    qpos_before = data.qpos[:7].copy()
    for _ in range(FRAME_SKIP):
        # PD control toward qpos_target
        data.ctrl[:7] = qpos_target.astype(np.float32)
        mujoco.mj_step(model, data)

    dq = float(np.linalg.norm((data.qpos[:7] - qpos_before)))
    motion_log.append(dq)

    # img0_chk = render_rgb(model, data, cam_id, width=224, height=224)
    joint_img0_chk = make_joint_observation(model, data, width=W, height=H)
    # joint_img.shape = (H, 2W, 3)

    # typical PyTorch transform (example)
    obs = joint_img0_chk.astype(np.float32) / 255.0
    obs = np.transpose(obs, (2, 0, 1))  # (C, H, W) = (3, H, 2W)
    obs = torch.from_numpy(obs)

    z0_chk = h_xi(preprocess_np_or_pil(joint_img0_chk, device=device)).detach()
    cos0 = torch.nn.functional.cosine_similarity(z0_chk, z_goal).item()
    print(f"[sanity] start cosine vs goal: {cos0:.4f}")

    EPS_EXPLORATION = 0.30   # try 0.15–0.30

    for step in range(args.steps):
        # Observe
        # img = render_rgb(model, data, cam_id, width=224, height=224)
        joint_img = make_joint_observation(model, data, width=W, height=H)
        # joint_img.shape = (H, 2W, 3)

        # typical PyTorch transform (example)
        obs = joint_img.astype(np.float32) / 255.0
        obs = np.transpose(obs, (2, 0, 1))  # (C, H, W) = (3, H, 2W)
        obs = torch.from_numpy(obs)


        if step < 30:  # first 30 control steps
            test_a = np.array([0.2, -0.15, 0.1, 0, 0, 0, 0], dtype=np.float32)

            qpos_before = data.qpos[:7].copy()

            qpos_target = qpos_target + test_a * (DT * FRAME_SKIP * ACTION_SCALE)
            qpos_target = np.clip(qpos_target, JOINT_MIN, JOINT_MAX)

            for _ in range(FRAME_SKIP):
                data.ctrl[:7] = qpos_target.astype(np.float32)
                mujoco.mj_step(model, data)

            dq = float(np.linalg.norm((data.qpos[:7] - qpos_before)))
            motion_log.append(dq)
            continue
        
        # Take a mid-run snapshot (e.g., step 1)
        if step == 100:
            Image.fromarray(joint_img).save("snapshots/step100.png")
            print("Saved snapshot → snapshots/step100.png")

        # Take a mid-run snapshot (e.g., step 100)
        if step == 1000:
            Image.fromarray(joint_img).save("snapshots/step1000.png")
            print("Saved snapshot → snapshots/step1000.png")

        if step == 10000:
            Image.fromarray(joint_img).save("snapshots/step10000.png")
            print("Saved snapshot → snapshots/step10000.png")

        # Take a mid-run snapshot (e.g., step 100)
        if step == 43100:
            Image.fromarray(joint_img).save("snapshots/step43100.png")
            print("Saved snapshot → snapshots/step43100.png")

        if step == 100000:
            Image.fromarray(joint_img).save("snapshots/step100000.png")
            print("Saved snapshot → snapshots/step100000.png")

        if step == 200000:
            Image.fromarray(joint_img).save("snapshots/step200000.png")
            print("Saved snapshot → snapshots/step200000.png")

        if step == 300000:
            Image.fromarray(joint_img).save("snapshots/step300000.png")
            print("Saved snapshot → snapshots/step300000.png")

        if step == 500000:
            Image.fromarray(joint_img).save("snapshots/step500000.png")
            print("Saved snapshot → snapshots/step500000.png")

        if step == 750000:
            Image.fromarray(joint_img).save("snapshots/step750000.png")
            print("Saved snapshot → snapshots/step750000.png")

        if step == 1000000:
            Image.fromarray(joint_img).save("snapshots/step1000000.png")
            print("Saved snapshot → snapshots/step1000000.png")

        if step % 1000 == 0:
            frames_out.append(joint_img.copy())  # store a copy of every 10 frames

        z_t  = encode_image(h_xi, joint_img, device=device).detach()

        # One-time probe (put before the loop or guarded by a flag)
        with torch.no_grad():
            z_probe = z_goal.clone()
            a_zeros = torch.zeros(1, Phi.cfg.a_dim, device=device)
            a_rand  = torch.randn(1, Phi.cfg.a_dim, device=device) * 0.3
            z1 = Phi(z_probe, a_zeros)
            z2 = Phi(z_probe, a_rand)
            diff = torch.norm(z2 - z1).item()

        if step % 1000 == 0:
            print(f"[Φ probe] ||Phi(z,a_rand)-Phi(z,0)|| = {diff:.6f}")


        # Plan one action sequence, execute first action
        a0 = cem_plan(
            z_t=z_t,
            z_goal=z_goal,
            Phi=Phi,
            act_low=ACT_LOW,
            act_high=ACT_HIGH,
            H=args.H,
            K=args.K,
            elites=args.elites,
            iters=args.iters
        )

        # --- Sanity: action magnitude ---
        a0_norm = float(np.linalg.norm(a0))
        if step % 1000 == 0:
            print(f"[{step:04d}] ||a0||={a0_norm:.4f}")

        # Epsilon-greedy exploration to break degeneracy
        if np.random.rand() < 0.50 or np.linalg.norm(a0) < 1e-3:
            a0 = a0 + np.random.uniform(ACT_LOW, ACT_HIGH).astype(np.float32) * EPS_EXPLORATION
        a0 = np.clip(a0, ACT_LOW, ACT_HIGH)


        # Execute via position actuators (integrate vel proposal → position target)
        qpos_before = data.qpos[:7].copy()

        qpos_target = qpos_target + a0 * (DT * FRAME_SKIP)   # integrate planned vel
        qpos_target = np.clip(qpos_target, JOINT_MIN, JOINT_MAX)

        for _ in range(FRAME_SKIP):
            data.ctrl[:7] = qpos_target.astype(np.float32)   # position targets
            mujoco.mj_step(model, data)

        # Log actual motion
        dq = float(np.linalg.norm((data.qpos[:7] - qpos_before)))
        motion_log.append(dq)
        if step % 1000 == 0:
            print(f"[{step:04d}] Δqpos={dq:.5f} rad")

        
        with torch.no_grad():
            # quick cost readout (roll 1 step to see direction)
            a_t = torch.from_numpy(a0[None, :]).to(device)
            z_pred = Phi(z_t, a_t)
            cost = torch.norm(z_pred - z_goal).item()
        if step % 1000 == 0:
            z_img = h_xi(preprocess_np_or_pil(joint_img, device=device)).detach()
            cos = torch.nn.functional.cosine_similarity(z_img, z_goal).item()
            print(f"[{step:04d}] cos(joint_img, goal)={cos:.5f}, cost≈{cost:.5f}, ||a0||={a0_norm:.4f}, dq={dq:.5f}")
        cost_log.append(cost)

    import json
    os.makedirs("logs", exist_ok=True)
    with open("logs/latent_costs.json", "w") as f:
        json.dump(cost_log, f)
    print("Final cost:", cost_log[-1])

    with open("logs/motion_log.json", "w") as f:
        json.dump(motion_log, f)
    print(f"Total arm displacement (sum of per-step dq): {sum(motion_log):.4f} rad")

    import imageio
    os.makedirs("videos", exist_ok=True)
    imageio.mimsave("videos/rollout.gif", frames_out, fps=15)
    print("Saved rollout video to videos/rollout.gif")

    print("Control loop finished.")

    # img_final = render_rgb(model, data, cam_id, width=224, height=224)
    joint_img_final = make_joint_observation(model, data, width=W, height=H)
    # joint_img.shape = (H, 2W, 3)

    # typical PyTorch transform (example)
    obs = joint_img_final.astype(np.float32) / 255.0
    obs = np.transpose(obs, (2, 0, 1))  # (C, H, W) = (3, H, 2W)
    obs = torch.from_numpy(obs)


    Image.fromarray(joint_img_final).save("snapshots/stepFinal.png")
    print("Saved final snapshot → snapshots/stepFinal.png")

    with torch.no_grad():
        z_final = h_xi(preprocess_np_or_pil(joint_img_final, device=device)).detach()
        z_goal = h_xi(preprocess_np_or_pil(goal_img, device=device)).detach()
        l2 = torch.norm(z_final - z_goal).item()
        cos = torch.nn.functional.cosine_similarity(z_final, z_goal).item()
        print(f"Final L2: {l2:.6f} | Cosine sim: {cos:.6f} | Norms: {z_final.norm().item():.3f}, {z_goal.norm().item():.3f}")

        if cos > 0.985:
            print("✅ Success: cosine similarity above threshold.")
        else:
            print("❌ Failure: cosine similarity below threshold.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_xml", required=True)
    ap.add_argument("--encoder_ckpt", required=True)
    ap.add_argument("--phi_ckpt", required=True)
    ap.add_argument("--goal_img", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--H", type=int, default=12)      # horizon
    ap.add_argument("--K", type=int, default=512)     # samples
    ap.add_argument("--elites", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    main(args)

