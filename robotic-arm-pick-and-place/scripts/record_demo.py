"""
scripts/record_demo.py  —  render the pick-and-place run to media/demo.gif.

Headless machine:  MUJOCO_GL=osmesa python scripts/record_demo.py
Normal Mac:        python scripts/record_demo.py   (no mjpython needed —
                   this renders offscreen, it does not open the live viewer)
"""

import sys, os
import mujoco
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.scripted_controller import ScriptedController
from utils.recording import Recorder

MODEL_PATH = "assets/franka_emika_panda/scene.xml"
TASKS = [("cube_green", "target_green"),
         ("cube_red", "target_red"),
         ("cube_blue", "target_blue")]


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    ctrl = ScriptedController(model, data)
    ctrl.reset()

    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
    rec = Recorder(model, width=640, height=480, camera=cam, fps=30, every=20)
    ctrl.recorder = rec

    ctrl._settle(300)
    for cube, target in TASKS:
        ok = ctrl.pick_and_place(cube, target)
        print(f"  {cube} -> {target}: {'SUCCESS' if ok else 'FAIL'}")

    os.makedirs("media", exist_ok=True)
    rec.save_gif("media/demo.gif")


if __name__ == "__main__":
    main()
