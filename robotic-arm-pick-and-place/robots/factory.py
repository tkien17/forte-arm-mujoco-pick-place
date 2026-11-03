"""Load robot scenes using repository-relative absolute paths."""

from pathlib import Path

import mujoco

from .forte import ForteRobot
from .panda import PandaRobot


PROJECT = Path(__file__).resolve().parents[1]
SCENES = {
    "forte": PROJECT / "assets" / "forte_arm" / "scene.xml",
    "panda": PROJECT / "assets" / "franka_emika_panda" / "scene.xml",
}


def load_robot(name="forte"):
    if name not in SCENES:
        raise ValueError(f"Unknown robot: {name}")
    model = mujoco.MjModel.from_xml_path(str(SCENES[name]))
    data = mujoco.MjData(model)
    robot = {"forte": ForteRobot, "panda": PandaRobot}[name](model, data)
    robot.reset()
    return robot
