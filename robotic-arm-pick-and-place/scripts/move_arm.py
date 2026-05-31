"""
scripts/move_arm.py  —  Phase 2 demo: smooth joint motion control.

Loads the pick-and-place scene (arm + table + 3 cubes + 3 targets), then
smoothly moves the arm through a few poses and opens/closes the gripper so
you can SEE controlled, non-jerky motion. No pick logic yet — that's Phase 3.

RUN WITH:   mjpython scripts/move_arm.py     (NOT python — macOS needs mjpython)
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

# --- Load the model -------------------------------------------------------
# Path is relative to where you run the command (the project root).
MODEL_PATH = "assets/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# The 7 arm-joint angles (radians) for a comfortable "home" pose.
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIPPER_OPEN = 255.0   # actuator8 control value: 255 = fully open
GRIPPER_CLOSE = 0.0    # 0 = fully closed


def reset():
    """Put cubes back on the table and the arm at its home pose."""
    mujoco.mj_resetData(model, data)   # restores XML spawn poses (cubes on table)
    data.qpos[:7] = HOME               # arm starts AT home (no startup swing)
    data.ctrl[:7] = HOME               # servo targets = home
    data.ctrl[7] = GRIPPER_OPEN        # gripper open
    mujoco.mj_forward(model, data)


def move_to(viewer, arm_target, gripper, seconds=1.5):
    """
    Smoothly drive the arm to `arm_target` (7 angles) over `seconds`.
    We ramp the actuator targets gradually instead of jumping — that's
    what makes the motion look smooth instead of snapping.
    """
    start = data.ctrl[:7].copy()
    arm_target = np.asarray(arm_target)
    n_steps = int(seconds / model.opt.timestep)
    for i in range(n_steps):
        if not viewer.is_running():
            return
        alpha = (i + 1) / n_steps               # goes 0 -> 1
        data.ctrl[:7] = start + alpha * (arm_target - start)  # linear blend
        data.ctrl[7] = gripper
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)          # run ~real-time


def main():
    reset()
    # A pose with the arm leaning forward over the table (hand-tuned).
    OVER_TABLE = np.array([0.0, 0.5, 0.0, -1.9, 0.0, 2.4, -0.785])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # let the cubes settle for half a second
        for _ in range(250):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # Loop the demo so you can watch the smooth motion repeatedly.
        while viewer.is_running():
            move_to(viewer, OVER_TABLE, GRIPPER_OPEN, seconds=2.0)  # lean in
            move_to(viewer, OVER_TABLE, GRIPPER_CLOSE, seconds=1.0) # close hand
            move_to(viewer, HOME, GRIPPER_CLOSE, seconds=2.0)       # back to home
            move_to(viewer, HOME, GRIPPER_OPEN, seconds=1.0)        # open hand


if __name__ == "__main__":
    main()
