"""Panda implementation of the shared interface, using its existing servos."""

import mujoco
import numpy as np

from .interface import RobotInterface


class PandaRobot(RobotInterface):
    def __init__(self, model, data):
        super().__init__(model, data, [f"joint{i}" for i in range(1, 8)],
                         [0, 0, 0, -1.57079, 0, 1.57079, -0.7853], [2.0] * 7)
        self.actuators = np.array([model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.gripper_actuator = model.actuator("actuator8").id
        self.velocity_to_position = (-model.actuator_biasprm[self.actuators, 2]
                                     / model.actuator_gainprm[self.actuators, 0])
        self.finger_positions = [int(model.joint(f"finger_joint{i}").qposadr[0]) for i in (1, 2)]
        self.finger_dofs = [int(model.joint(f"finger_joint{i}").dofadr[0]) for i in (1, 2)]
        self.effort_limits = model.actuator_forcerange[self.actuators, 1].copy()
        self.open_width, self.closed_width = 0.08, 0.0
        self.set_gripper_width(self.open_width)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.home
        self.data.qpos[self.finger_positions] = self.open_width / 2
        self.set_arm_target(self.home)
        self.set_gripper_width(self.open_width)
        for color in ("red", "green", "blue"):
            joint = int(self.model.body(f"cube_{color}").jntadr[0])
            self.data.qpos[self.model.jnt_qposadr[joint] + 2] = 0.22
        self._apply_control()
        mujoco.mj_forward(self.model, self.data)

    def set_gripper_width(self, width, driver_speed=0.0):
        if not np.isfinite(width) or not 0 <= width <= self.open_width:
            raise ValueError("Panda opening must be between 0 and 0.08 m")
        if not np.isfinite(driver_speed) or driver_speed != 0:
            raise ValueError("Panda servo backend accepts position targets with zero speed feedforward")
        self.gripper_target = float(width)

    def _apply_control(self):
        self.data.qfrc_applied[:] = 0
        # The existing servo implements kp*(ctrl-q)-kd*dq. Offset its input
        # to express the shared interface's optional velocity reference.
        command = self.target_q + self.velocity_to_position * self.target_dq
        control_limits = self.model.actuator_ctrlrange[self.actuators]
        self.data.ctrl[self.actuators] = np.clip(command, control_limits[:, 0], control_limits[:, 1])
        self.data.ctrl[self.gripper_actuator] = 255 * self.gripper_target / self.open_width

    def step(self):
        super().step()
        self.last_effort = self.data.qfrc_actuator[self.dof_indices].copy()
        self.last_raw_effort = self.last_effort.copy()
        self.last_gripper_effort = float(self.data.actuator_force[self.gripper_actuator])
