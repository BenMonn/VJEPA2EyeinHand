import mujoco
import mujoco.viewer

# Path to the merged XML file
model_path = "/home/bmonn/vjepa/menagerie/franka_emika_panda/mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(model_path)
print(model)

# Create data associated with the model
data = mujoco.MjData(model)

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer launched. Press ESC to exit.")
    # Run until user closes the window
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
