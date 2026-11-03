"""Full-pose damped-least-squares IK on scratch state, with explicit failure.

Only the robot's named hinge DOFs enter the solve. Finger and object coordinates
are never decision variables. Endpoint screening does not replace path checks.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    iterations: int
    reason: str


def rotation_error(desired, current):
    inverse, difference, velocity = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(inverse, current)
    mujoco.mju_mulQuat(difference, desired, inverse)
    if difference[0] < 0:
        difference *= -1
    mujoco.mju_quat2Vel(velocity, difference, 1.0)
    return velocity


def downward_quaternion(yaw=0.0):
    """Map the Forte tool's +X to world -Z, and choose closing-axis yaw."""
    pitch = np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])
    turn = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, turn, pitch)
    return result


class PoseIK:
    def __init__(self, robot, position_tolerance=0.0005, rotation_tolerance=np.deg2rad(0.5),
                 joint_margin=0.01, torque_fraction=0.90, environment_clearance=0.002):
        self.robot, self.model = robot, robot.model
        self.data = mujoco.MjData(self.model)
        self.position_tolerance, self.rotation_tolerance = position_tolerance, rotation_tolerance
        self.joint_margin, self.torque_fraction = joint_margin, torque_fraction
        self.environment_clearance = environment_clearance
        environment = {0, *(self.model.body(f"cube_{c}").id for c in ("red", "green", "blue"))}
        table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
        if table >= 0:
            environment.add(table)
        self.robot_geoms = np.array([body not in environment for body in self.model.geom_bodyid])
        self.moving_collision_geoms = np.flatnonzero(self.robot_geoms & (self.model.geom_contype != 0))
        self.environment_geoms = [self.model.geom(name).id for name in
                                  ("floor", "table_top", "cube_red", "cube_green", "cube_blue")]
        self.last_clearance = float("inf")
        self.jp, self.jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.payload = None

    def set_payload(self, body_name):
        """Move a held object only in scratch planning, never in live physics."""
        self.payload = None
        if body_name is None:
            return
        live = self.robot.data
        body = self.model.body(body_name).id
        site = self.robot.site_id
        rotation = live.site_xmat[site].reshape(3, 3)
        tool_quat, inverse, relative = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(tool_quat, live.site_xmat[site])
        mujoco.mju_negQuat(inverse, tool_quat)
        mujoco.mju_mulQuat(relative, inverse, live.xquat[body])
        self.payload = {"body": body, "geom": self.model.geom(body_name).id,
                        "qpos": int(self.model.jnt_qposadr[self.model.body_jntadr[body]]),
                        "offset": rotation.T @ (live.xpos[body] - live.site_xpos[site]),
                        "quaternion": relative}

    def allowed_payload_contact(self, a, b):
        if self.payload is None or self.payload["geom"] not in (a, b):
            return False
        other = b if a == self.payload["geom"] else a
        return self.model.geom(other).name.startswith(("l_finger_finger_", "r_finger_finger_"))

    def pose(self, q):
        self.data.qpos[:] = self.robot.data.qpos
        self.data.qpos[self.robot.qpos_indices] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        if self.payload is not None:
            p = self.payload
            rotation = self.data.site_xmat[self.robot.site_id].reshape(3, 3)
            position = self.data.site_xpos[self.robot.site_id] + rotation @ p["offset"]
            tool_quat, quaternion = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(tool_quat, self.data.site_xmat[self.robot.site_id])
            mujoco.mju_mulQuat(quaternion, tool_quat, p["quaternion"])
            self.data.qpos[p["qpos"]:p["qpos"] + 7] = np.r_[position, quaternion]
            mujoco.mj_forward(self.model, self.data)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self.robot.site_id])
        return self.data.site_xpos[self.robot.site_id].copy(), quaternion

    def screen(self, q):
        self.last_clearance = float("inf")
        self.pose(q)
        margin = float(np.min(np.r_[q - self.robot.lower, self.robot.upper - q]))
        gravity = self.data.qfrc_bias[self.robot.dof_indices].copy()
        if self.payload is not None:
            body = self.payload["body"]
            mujoco.mj_jac(self.model, self.data, self.jp, None, self.data.xipos[body],
                          int(self.model.site_bodyid[self.robot.site_id]))
            gravity += self.jp[:, self.robot.dof_indices].T @ (-self.model.body_mass[body] * self.model.opt.gravity)
        fraction = float(np.max(np.abs(gravity) / self.robot.effort_limits))
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (self.robot_geoms[a] or self.robot_geoms[b]) and contact.dist < -1e-5 and not self.allowed_payload_contact(a, b):
                return False, f"collision: {self.model.geom(a).name}/{self.model.geom(b).name}", margin, fraction
        if margin < self.joint_margin - 1e-9:
            return False, "joint-limit margin", margin, fraction
        if fraction > self.torque_fraction:
            return False, "static gravity torque margin", margin, fraction
        self.last_clearance = self.environment_distance()
        if self.last_clearance < self.environment_clearance:
            return False, "environment clearance", margin, fraction
        return True, "ok", margin, fraction

    def environment_distance(self):
        """Minimum moving-robot/environment distance, capped at 20 mm."""
        distance = 0.02
        for a in self.moving_collision_geoms:
            for b in self.environment_geoms:
                if self.payload is not None and b == self.payload["geom"] and self.allowed_payload_contact(int(a), b):
                    continue
                distance = min(distance, mujoco.mj_geomDistance(
                    self.model, self.data, int(a), b, distance, None))
                if distance < self.environment_clearance:
                    return float(distance)
        if self.payload is not None:
            held = self.payload["geom"]
            for b in self.environment_geoms:
                if b == held:
                    continue
                separation = mujoco.mj_geomDistance(self.model, self.data, held, b, 0.02, None)
                # Initial lift starts from supported contact; no table penetration.
                threshold = -0.0005 if self.model.geom(b).name in ("floor", "table_top") else self.environment_clearance
                if separation < threshold:
                    return -1.0
        return float(distance)

    def solve(self, target_position, target_quaternion, q_seed, max_iterations=250):
        target = np.asarray(target_position, dtype=float)
        desired = np.asarray(target_quaternion, dtype=float)
        q = np.asarray(q_seed, dtype=float).copy()
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("IK position must be a finite 3-vector")
        if desired.shape != (4,) or not np.isfinite(desired).all() or np.linalg.norm(desired) < 1e-12:
            raise ValueError("IK orientation must be a finite nonzero wxyz quaternion")
        if q.shape != self.robot.home.shape or not np.isfinite(q).all():
            raise ValueError("IK seed must have one finite entry per arm joint")
        desired = desired / np.linalg.norm(desired)
        low, high = self.robot.lower + self.joint_margin, self.robot.upper - self.joint_margin
        q = np.clip(q, low, high)
        reason = "iteration limit / unreachable from seed"
        for iteration in range(max_iterations + 1):
            position, current = self.pose(q)
            ep, er = target - position, rotation_error(desired, current)
            pe, re = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if pe <= self.position_tolerance and re <= self.rotation_tolerance:
                accepted, reason, _, _ = self.screen(q)
                return IKResult(accepted, q.copy(), pe, re, iteration, reason)
            if iteration == max_iterations:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.robot.site_id)
            jacobian = np.vstack([self.jp[:, self.robot.dof_indices], 0.2 * self.jr[:, self.robot.dof_indices]])
            error = np.r_[ep, 0.2 * er]
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.005**2 * np.eye(6), error)
            dq *= min(1, 0.15 / max(float(np.max(np.abs(dq))), 1e-12))
            score = float(error @ error)
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = np.clip(q + scale * dq, low, high)
                cp, cq = self.pose(candidate)
                ce = np.r_[target - cp, 0.2 * rotation_error(desired, cq)]
                if float(ce @ ce) < score - 1e-14:
                    q = candidate
                    improved = True
                    break
            if not improved:
                reason = "stalled at joint limit / unreachable from seed"
                break
        return IKResult(False, q.copy(), pe, re, iteration, reason)
