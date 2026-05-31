"""
scripts/run_pick_place.py  —  Phase 3 demo.

Loads the scene, then picks up each cube and places it on the matching
colored target pad, printing success/failure for each.

RUN WITH:   mjpython scripts/run_pick_place.py     (NOT python)
"""

import sys, os
import numpy as np
import mujoco
import mujoco.viewer

# make the project root importable so we can load controllers/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.scripted_controller import ScriptedController

MODEL_PATH = "assets/franka_emika_panda/scene.xml"

# Which cube goes on which target pad.
TASKS = [
    ("cube_green", "target_green"),
    ("cube_red",   "target_red"),
    ("cube_blue",  "target_blue"),
]


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    ctrl = ScriptedController(model, data)
    ctrl.reset()

    results = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        ctrl._settle(300, viewer)   # let the cubes settle on the table
        for cube, target in TASKS:
            if not viewer.is_running():
                break
            ok = ctrl.pick_and_place(cube, target, viewer)
            results.append((cube, ok))
            print(f"  {cube:11s} -> {target:13s} : {'SUCCESS ✅' if ok else 'FAILED ❌'}")

        # report and keep the window open to admire the result
        n_ok = sum(1 for _, ok in results if ok)
        print(f"\nDone: {n_ok}/{len(results)} placed successfully.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
