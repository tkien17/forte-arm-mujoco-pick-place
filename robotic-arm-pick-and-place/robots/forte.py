"""Six-joint Forte arm: bounded PD + pure gravity feedforward.

Gravity is evaluated at the measured pose on a zero-velocity scratch state.
Only arm gravity terms are applied; free objects receive no compensating force.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .forte_gripper import ForteGripper
from .interface import RobotInterface


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs" / "forte.json"
URDF_PATH = PROJECT / "assets" / "forte_arm" / "source" / "ForteME.urdf"


class ForteRobot(RobotInterface):
    def __init__(self, model, data):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["arm"]
        names = tuple(self.config["joints"])
        entries = list(self.config["joints"].values())
        source = ET.parse(URDF_PATH).getroot()
        limits = [source.find(f"joint[@name='{name}']/limit").attrib for name in names]
        super().__init__(model, data, names, [item["home_rad"] for item in entries],
                         [float(limit["velocity"]) for limit in limits])
        if len(names) != 6 or model.nu != 0 or model.neq != 1:
            raise ValueError("Expected the unbraced Forte task scene: 6 arm joints, no actuators, one mimic")
        if np.any(model.body_gravcomp != 0):
            raise ValueError("Passive gravity compensation would double-count Forte feedforward")
        self.kp = np.array([item["kp_nm_per_rad"] for item in entries], dtype=float)
        self.kd = np.array([item["kd_nm_s_per_rad"] for item in entries], dtype=float)
        self.effort_limits = np.array([float(limit["effort"]) for limit in limits])
        for vector in (self.kp, self.kd, self.effort_limits, self.speed_limits):
            if not np.isfinite(vector).all() or np.any(vector <= 0):
                raise ValueError("Forte gains and effort/speed limits must be finite and positive")
        np.testing.assert_allclose(model.jnt_actfrcrange[self.joint_ids, 1], self.effort_limits)
        self.gripper = ForteGripper(model)
        self.open_width = self.gripper.config["open_width_m"]
        self.closed_width = self.gripper.config["empty_close_width_m"]
        self.gravity_data = mujoco.MjData(model)
        self.payload_body = None
        self.payload_jacobian = np.zeros((3, model.nv))
        self.set_gripper_width(self.open_width)

    def reset(self):
        self.set_payload(None)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.qpos[self.qpos_indices] = self.home
        self.gripper.initialize(self.data, self.open_width)
        self.set_arm_target(self.home)
        self.set_gripper_width(self.open_width)
        mujoco.mj_forward(self.model, self.data)

    def set_gripper_width(self, width, driver_speed=0.0):
        self.gripper.position_from_width(width)
        if not np.isfinite(driver_speed) or abs(driver_speed) > self.gripper.config["speed_limit_m_per_s"]:
            raise ValueError("Gripper target speed exceeds its limit")
        self.gripper_target = float(width)
        self.gripper_speed = float(driver_speed)

    def gravity_effort(self):
        self.gravity_data.qpos[:] = self.data.qpos
        self.gravity_data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.gravity_data)
        effort = self.gravity_data.qfrc_bias[self.dof_indices].copy()
        if self.payload_body is not None:
            point = self.gravity_data.xipos[self.payload_body]
            mujoco.mj_jac(self.model, self.gravity_data, self.payload_jacobian, None,
                          point, int(self.model.site_bodyid[self.site_id]))
            support_force = -self.model.body_mass[self.payload_body] * self.model.opt.gravity
            effort += self.payload_jacobian[:, self.dof_indices].T @ support_force
        return effort

    def set_payload(self, body_name):
        """Compensate a grasped object's weight; this adds no attachment constraint."""
        self.payload_body = None if body_name is None else self.model.body(body_name).id

    def _apply_control(self):
        self.last_gravity = self.gravity_effort()
        self.last_raw_effort = (self.kp * (self.target_q - self.arm_position)
                               + self.kd * (self.target_dq - self.arm_velocity)
                               + self.last_gravity)
        self.last_effort = np.clip(self.last_raw_effort, -self.effort_limits, self.effort_limits)
        self.data.qfrc_applied[:] = 0
        self.data.qfrc_applied[self.dof_indices] = self.last_effort
        self.last_gripper_effort = self.gripper.apply(self.data, self.gripper_target, self.gripper_speed)
