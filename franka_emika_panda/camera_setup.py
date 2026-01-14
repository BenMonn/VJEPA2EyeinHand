# camera_setup.py
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

MODEL_XML = "/home/bmonn/vjepa/menagerie/franka_emika_panda/arm_scene.xml"  # your Panda+table+object scene

W = 224
H = 224

_renderer_cache = {}  # (model.ptr, w, h) -> Renderer

def get_cam_id(model, cam_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in model XML.")
    return cam_id

def render_rgb(model, data, cam_id, width=224, height=224):
    """
    Offscreen render using mujoco.Renderer (manages GL context internally).
    """
    key = (id(model), width, height)
    renderer = _renderer_cache.get(key)
    if renderer is None:
        renderer = mujoco.Renderer(model, width, height)
        _renderer_cache[key] = renderer

    # Update the internal scene from the current data and camera
    renderer.update_scene(data, camera=cam_id)
    # Render returns uint8 RGB image of shape (H,W,3)
    rgb = renderer.render()
    # Defensive copy so cached renderer's buffer isn't mutated outside
    return np.array(rgb, copy=True)

def render_two_cameras(model, data, width=W, height=H):
    """
    Returns: (img_global, img_eye)
    each: (H, W, 3) uint8
    """
    cam_global = get_cam_id(model, "global")
    cam_eye = get_cam_id(model, "eye_in_hand")

    img_global = render_rgb(model, data, cam_global, width, height)
    img_eye = render_rgb(model, data, cam_eye, width, height)

    return img_global, img_eye

def make_joint_observation(model, data, width=W, height=H):
    img_global, img_eye = render_two_cameras(model, data, width, height)
    # side-by-side panorama: (H, 2W, 3)
    joint = np.concatenate([img_global, img_eye], axis=1)
    return joint  # uint8, ready for saving or feeding into your dataset

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)

    # --- sanity check: individual cameras ---
    img_global, img_eye = render_two_cameras(model, data, width=W, height=H)
    Image.fromarray(img_global).save("sanity_global.png")
    Image.fromarray(img_eye).save("sanity_eye.png")

    # --- sanity check: joint panorama for V-JEPA ---
    joint = np.concatenate([img_global, img_eye], axis=1)  # (H, 2W, 3)
    Image.fromarray(joint).save("sanity_joint.png")

    print("Saved sanity_global.png, sanity_eye.png, sanity_joint.png")

