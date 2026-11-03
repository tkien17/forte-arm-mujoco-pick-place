"""Common joint-reference interface; every step recomputes robot control.

Arm targets use radians and radians/second. Gripper opening uses metres.
Backends own ctrl/qfrc_applied; xfrc_applied remains available for external loads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import mujoco
import numpy as np


class RobotInterface(ABC):
    def __init__(self, model, data, joint_names, home, speed_limits):
        self.model, self.data = model, data
        self.joint_names = tuple(joint_names)
        self.joint_ids = np.array([model.joint(name).id for name in joint_names])
        if np.any(model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Expected one hinge coordinate per arm joint")
        self.qpos_indices = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_indices = model.jnt_dofadr[self.joint_ids].copy()
        self.lower, self.upper = model.jnt_range[self.joint_ids].T.copy()
        self.home = np.asarray(home, dtype=float)
        self.speed_limits = np.asarray(speed_limits, dtype=float)
        self.target_q = self.home.copy()
        self.target_dq = np.zeros(len(joint_names))
        self.site_id = model.site("grasp_site").id
        self.last_effort = np.zeros(len(joint_names))
        self.last_raw_effort = np.zeros(len(joint_names))
        self.last_gravity = np.zeros(len(joint_names))
        self.last_gripper_effort = 0.0
        self.set_arm_target(self.home)

    def set_arm_target(self, position, velocity=None):
        position = np.asarray(position, dtype=float)
        velocity = np.zeros(len(self.joint_names)) if velocity is None else np.asarray(velocity, dtype=float)
        for vector in (position, velocity):
            if vector.shape != self.home.shape or not np.isfinite(vector).all():
                raise ValueError("Arm targets must be finite vectors with one entry per arm joint")
        if np.any(position < self.lower) or np.any(position > self.upper):
            raise ValueError("Arm target exceeds joint limits")
        if np.any(np.abs(velocity) > self.speed_limits):
            raise ValueError("Arm target velocity exceeds joint limits")
        self.target_q, self.target_dq = position.copy(), velocity.copy()

    @property
    def arm_position(self):
        return self.data.qpos[self.qpos_indices].copy()

    @property
    def arm_velocity(self):
        return self.data.qvel[self.dof_indices].copy()

    @property
    def tool_position(self):
        return self.data.site_xpos[self.site_id].copy()

    @abstractmethod
    def reset(self):
        """Initialize the episode; direct position assignment is allowed here only."""

    @abstractmethod
    def set_gripper_width(self, width, driver_speed=0.0):
        pass

    @abstractmethod
    def _apply_control(self):
        pass

    def step(self):
        """Use this for moves, settling, dwell, recording, and viewer idle."""
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite robot state before control")
        self._apply_control()
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or any(w.number for w in self.data.warning)):
            raise RuntimeError("Nonfinite state or MuJoCo numerical warning after stepping")
