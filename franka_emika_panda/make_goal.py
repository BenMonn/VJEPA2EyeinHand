# make_goal.py
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np, mujoco
from PIL import Image
from camera_setup import make_joint_observation

MODEL_XML = "/home/bmonn/vjepa/menagerie/franka_emika_panda/arm_scene.xml"
OUT_PATH  = "goal.png"
W = H = 224


def hold_current_targets(model, data, steps=120):
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        data.ctrl[:7] = data.qpos[:7].astype(np.float32)
        mujoco.mj_step(model, data)


def main():
    # load model, reset
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # === Put arm in desired GOAL configuration ===
    goal_qpos = np.array([ 0.35, -1.0, 0.25, -1.8, 0.15, 1.6, 0.9 ], dtype=np.float32)
    data.qpos[:7] = goal_qpos
    hold_current_targets(model, data, steps=160)

    # === Capture the MULTIVIEW goal image ===
    joint_img = make_joint_observation(model, data, width=W, height=H)

    # === Save goal ===
    Image.fromarray(joint_img).save(OUT_PATH)
    print(f"Saved multiview goal → {OUT_PATH}")


if __name__ == "__main__":
    main()

