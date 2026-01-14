# collect_dataset.py
import os, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

print("MUJOCO_GL =", os.getenv("MUJOCO_GL"))
# Minimal empty model is enough to test the GL context:
model = mujoco.MjModel.from_xml_string("<mujoco/>")
data  = mujoco.MjData(model)

renderer = mujoco.Renderer(model, 64, 64)
renderer.update_scene(data)      # default free camera
img = renderer.render()          # (64,64,3) uint8
print("Render OK, shape:", np.asarray(img).shape)

from PIL import Image, ImageEnhance, ImageFilter
from camera_setup import make_joint_observation, get_cam_id
MODEL_XML = "/home/bmonn/vjepa/menagerie/franka_emika_panda/arm_scene.xml"
OUT_DIR = "/home/bmonn/vjepa/menagerie/franka_emika_panda/dataset"
N_EPISODES = 500
H = 100              # steps per episode
FPS = 15
WIDTH = HEIGHT = 224

os.makedirs(OUT_DIR, exist_ok=True)

def random_vel(k=7, bound=0.6, std_frac=0.25):
    std = bound * std_frac
    v = np.random.randn(k).astype(np.float32) * std
    return np.clip(v, -bound, bound)

model = mujoco.MjModel.from_xml_path(MODEL_XML)
data = mujoco.MjData(model)
cam_global = get_cam_id(model, "global")
cam_eye    = get_cam_id(model, "eye_in_hand")

import random

def _rand_u(a, b):  # uniform helper
    return np.random.uniform(a, b)

def randomize_episode(model, data, rng=np.random):
    """
    Apply reasonable domain randomizations in-place.
    Call this once per episode, and optionally light/camera jitter every N steps.
    """
    # ----- Colors / textures (simple: geom rgba) -----
    # Randomize a few named geoms if you have them; otherwise broadcast to groups.
    # Example: darken/brighten all geoms slightly.
    if hasattr(model, "geom_rgba") and model.geom_rgba.size > 0:
        rgba = model.geom_rgba.copy() # scale brightness and saturation a bit 
        brightness = rng.uniform(0.7, 1.3) 
        tint = rng.uniform(0.9, 1.1, size=(rgba.shape[0], 3)) 
        rgba[:, :3] = np.clip(rgba[:, :3] * tint * brightness, 0.0, 1.0) 
        model.geom_rgba[:] = rgba

    # ----- Lights -----
    # If your model has lights, randomize their position and ambient/diffuse.
    if getattr(model, "nlight", 0) > 0:
        for lid in range(model.nlight):
            if hasattr(model, "light_pos"):
                model.light_pos[lid, :] += rng.uniform(-0.1, 0.1, size=3)
            if hasattr(model, "light_dir"):
                model.light_dir[lid, :] += rng.uniform(-0.05, 0.05, size=3)
            if hasattr(model, "light_ambient"):
                model.light_ambient[lid, :] = np.clip(
                    model.light_ambient[lid, :] * rng.uniform(0.7, 1.3, size=3), 0.0, 1.0
                )
            if hasattr(model, "light_diffuse"):
                model.light_diffuse[lid, :] = np.clip(
                    model.light_diffuse[lid, :] * rng.uniform(0.7, 1.3, size=3), 0.0, 1.0
                )

    # ----- Camera pose (per-episode) -----
    # Small perturbation to camera to force viewpoint robustness.
        # ----- Camera pose (per-episode) -----
    if hasattr(model, "cam_pos"):
        for cam_name in ("global", "eye_in_hand"):
            try:
                cid = get_cam_id(model, cam_name)
            except ValueError:
                continue
            model.cam_pos[cid, :] += rng.uniform(-0.015, 0.015, size=3)

    if hasattr(model, "cam_quat"):
        for cam_name in ("global", "eye_in_hand"):
            try:
                cid = get_cam_id(model, cam_name)
            except ValueError:
                continue
            jitter = rng.uniform(-0.02, 0.02, size=4)
            q = model.cam_quat[cid, :].copy()
            q = q / (np.linalg.norm(q) + 1e-9)
            q = q + jitter
            q = q / (np.linalg.norm(q) + 1e-9)
            model.cam_quat[cid, :] = q

    # ----- Physics: friction, damping, mass (moderate ranges) -----
    # Friction: scale all geoms’ slide friction a bit
    if hasattr(model, "geom_friction"):
        scale = rng.uniform(0.7, 1.3)
        model.geom_friction[:, 0] = np.clip(model.geom_friction[:, 0] * scale, 0.1, 5.0)

    # DOF damping (joint viscous damping)
    if hasattr(model, "dof_damping"):
        model.dof_damping[:] = np.clip(
            model.dof_damping[:] * rng.uniform(0.8, 1.2), 0.001, 500.0
        )

    # Link masses (very careful—small changes)
    if hasattr(model, "body_mass"):
        model.body_mass[:] = np.clip(
            model.body_mass[:] * rng.uniform(0.95, 1.05), 1e-3, 1e3
        )

    # Re‐init data after parameter edits
    mujoco.mj_resetData(model, data)


# Optional per-frame imaging noise (sensor-space)
def augment_frame_np(img_np, rng=np.random):
    """
    img_np: uint8 (H,W,3)
    Returns uint8 (H,W,3) with mild photometric noise.
    """
    im = Image.fromarray(img_np)

    # Random brightness / contrast / saturation / sharpness
    if random.random() < 0.9:
        im = ImageEnhance.Brightness(im).enhance(_rand_u(0.7, 1.3))
    if random.random() < 0.9:
        im = ImageEnhance.Contrast(im).enhance(_rand_u(0.7, 1.3))
    if random.random() < 0.7:
        im = ImageEnhance.Color(im).enhance(_rand_u(0.7, 1.4))
    if random.random() < 0.5:
        im = ImageEnhance.Sharpness(im).enhance(_rand_u(0.8, 1.2))

    # Small Gaussian blur sometimes
    if random.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(radius=_rand_u(0.0, 1.2)))

    # Add sensor noise (Gaussian)
    arr = np.asarray(im).astype(np.float32)
    if random.random() < 0.8:
        arr += rng.normal(0.0, _rand_u(0.0, 8.0), size=arr.shape)  # 0..8 intensity
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)

for ep in range(N_EPISODES):
    mujoco.mj_resetData(model, data)

    randomize_episode(model, data)

    frames = []
    acts = []
    for t in range(H):
        if t % 20 == 0 and hasattr(model, "cam_pos"):
            # small per-frame camera jitter on both cams
            for cam_name in ("global", "eye_in_hand"):
                try:
                    cid = get_cam_id(model, cam_name)
                except ValueError:
                    continue
                model.cam_pos[cid, :] += np.random.uniform(-0.005, 0.005, size=3)

        # random wiggles (replace later with scripted pick/place)
        qv = random_vel(k=min(7, model.nv))
        acts.append(qv)
        data.ctrl[:] = 0.0
        data.qvel[:len(qv)] = qv
        mujoco.mj_step(model, data)
        if t % int(model.opt.timestep * FPS / model.opt.timestep) == 0:
            img = make_joint_observation(model, data, width=WIDTH, height=HEIGHT)
            img = augment_frame_np(img)
            frames.append(img)

    clip = np.stack(frames)  # [T, H, W, 3]
    acts = np.stack(acts)   # [T, model.nu]
    np.savez(os.path.join(OUT_DIR, f"ep_{ep:05d}.npz"),
            frames=clip,
            actions=acts)
    print(f"Saved ep_{ep:05d}.npz with {len(frames)} frames")
