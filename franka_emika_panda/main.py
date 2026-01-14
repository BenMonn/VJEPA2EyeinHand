import argparse, torch
from sim.pick_and_place import pick_and_place
from sim.calibration import run_calibration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calibrate","run"], default="run")
    args = parser.parse_args()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    run_calibration("arm_scene.xml")
    pick_and_place("arm_scene.xml")

    if args.stage=="calibrate":
        run_calibration("arm_scene.xml")
    else:
        pick_and_place()