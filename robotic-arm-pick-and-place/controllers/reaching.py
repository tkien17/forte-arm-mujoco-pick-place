"""Screened Forte reaching paths, with optional held-object planning."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .ik import PoseIK, downward_quaternion


WORKSPACE_PATH = Path(__file__).resolve().parents[1] / "configs" / "forte.json"


@dataclass
class JointPath:
    label: str
    nodes: np.ndarray
    duration: float
    desired_position: np.ndarray
    desired_quaternion: np.ndarray
    cartesian: bool = False

    def sample(self, time):
        fraction = float(np.clip(time / self.duration, 0, 1))
        blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        rate = (30 * fraction**2 - 60 * fraction**3 + 30 * fraction**4) / self.duration
        scaled = blend * (len(self.nodes) - 1)
        index = min(int(scaled), len(self.nodes) - 2)
        alpha = scaled - index
        q = self.nodes[index] + alpha * (self.nodes[index + 1] - self.nodes[index])
        dq = (self.nodes[index + 1] - self.nodes[index]) * (len(self.nodes) - 1) * rate
        return q, dq


class ReachPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.config = json.loads(WORKSPACE_PATH.read_text(encoding="utf-8"))["workspace"]
        np.testing.assert_allclose(robot.model.geom("table_top").pos, self.config["table_center_m"])
        np.testing.assert_allclose(robot.model.geom("table_top").size, self.config["table_half_size_m"])
        for color in ("red", "green", "blue"):
            np.testing.assert_allclose(robot.data.site(f"target_{color}").xpos[:2],
                                       self.config["target_xy_m"][color])
        self.ik = PoseIK(robot, joint_margin=self.config["joint_margin_rad"],
                         torque_fraction=self.config["static_torque_fraction"],
                         environment_clearance=self.config["environment_clearance_m"])
        self.minimum_joint_margin = float("inf")
        self.minimum_environment_clearance = float("inf")
        self.maximum_static_fraction = 0.0
        self.checked_samples = 0

    def check_q(self, q):
        accepted, reason, margin, fraction = self.ik.screen(q)
        self.checked_samples += 1
        if not accepted:
            raise ValueError(f"Reference rejected before execution: {reason}")
        self.minimum_joint_margin = min(self.minimum_joint_margin, margin)
        self.maximum_static_fraction = max(self.maximum_static_fraction, fraction)
        self.minimum_environment_clearance = min(self.minimum_environment_clearance, self.ik.last_clearance)

    def joint_path(self, label, start, end, quaternion, position):
        count = max(2, int(np.ceil(np.max(np.abs(end - start)) /
                                   self.config["joint_check_spacing_rad"])) + 1)
        nodes = np.linspace(start, end, count)
        return self.finish_path(label, nodes, quaternion, position, False)

    def cartesian_path(self, label, start_q, end_position, quaternion):
        start_position, _ = self.ik.pose(start_q)
        count = max(2, int(np.ceil(np.linalg.norm(end_position - start_position) /
                                   self.config["cartesian_spacing_m"])) + 1)
        nodes = [start_q.copy()]
        for point in np.linspace(start_position, end_position, count)[1:]:
            solved = self.ik.solve(point, quaternion, nodes[-1])
            if not solved.success:
                raise ValueError(f"{label}: IK failed: {solved.reason}")
            if np.max(np.abs(solved.q - nodes[-1])) > 0.15:
                raise ValueError(f"{label}: IK branch discontinuity")
            nodes.append(solved.q)
        return self.finish_path(label, np.array(nodes), quaternion, end_position, True)

    def finish_path(self, label, nodes, quaternion, position, cartesian):
        # A quintic global time law starts/stops at zero speed and acceleration.
        slopes = np.diff(nodes, axis=0) * (len(nodes) - 1)
        duration = max(3.0, float(1.875 * np.max(np.abs(slopes)) /
                                  self.config["reference_speed_rad_per_s"]))
        path = JointPath(label, nodes, duration, np.asarray(position).copy(), quaternion.copy(), cartesian)
        for a, b in zip(nodes[:-1], nodes[1:]):
            steps = max(2, int(np.ceil(np.max(np.abs(b - a)) /
                                       self.config["joint_check_spacing_rad"])) + 1)
            for q in np.linspace(a, b, steps):
                self.check_q(q)
        # Check the actual time-parameterized reference, including interpolated nodes.
        for t in np.linspace(0, duration, max(101, len(nodes) * 2)):
            q, dq = path.sample(t)
            if np.any(np.abs(dq) > self.robot.speed_limits):
                raise ValueError(f"{label}: reference speed exceeds URDF limits")
            self.check_q(q)
        return path

    def seeds(self, position):
        yield self.robot.home
        for roll, wrist in ((0, 1.2), (-2.9, -1.3), (2.9, -1.3)):
            values = dict(zip(("shoulder_yaw", "shoulder_pitch", "shoulder_roll", "elbow_pitch",
                               "lower_arm_roll", "wrist_pitch"),
                              (np.arctan2(position[1], position[0]), -1.2, 0, 1.6, roll, wrist)))
            yield np.array([values[name] for name in self.robot.joint_names])

    def plan_task(self, color):
        robot = self.robot
        cube = robot.data.body(f"cube_{color}").xpos.copy()
        target = robot.data.site(f"target_{color}").xpos.copy()
        half = float(robot.model.geom(f"cube_{color}").size[2])
        table_z = self.config["table_center_m"][2] + self.config["table_half_size_m"][2]
        cube_hover = np.array([cube[0], cube[1], self.config["hover_z_m"]])
        target_hover = np.array([target[0], target[1], self.config["hover_z_m"]])
        release = np.array([target[0], target[1], table_z + half + self.config["release_clearance_m"]])
        reasons = []
        for yaw in self.config["yaw_candidates_deg"]:
            quaternion = downward_quaternion(np.deg2rad(yaw))
            for seed in self.seeds(cube_hover):
                solution = self.ik.solve(cube_hover, quaternion, seed)
                if not solution.success:
                    reasons.append(solution.reason)
                    continue
                try:
                    paths = [self.joint_path(f"{color}: approach above cube", robot.home, solution.q,
                                             quaternion, cube_hover)]
                    for label, point in (("descend around cube (OPEN)", cube),
                                         ("retreat above cube", cube_hover),
                                         ("travel above target", target_hover),
                                         ("lower above target (OPEN)", release),
                                         ("retreat above target", target_hover)):
                        paths.append(self.cartesian_path(f"{color}: {label}", paths[-1].nodes[-1], point, quaternion))
                    home_position, home_quaternion = self.ik.pose(robot.home)
                    paths.append(self.joint_path(f"{color}: return home", paths[-1].nodes[-1], robot.home,
                                                 home_quaternion, home_position))
                    return paths, {"color": color, "yaw_deg": yaw, "cube_m": cube.tolist(),
                                   "target_m": target.tolist(), "hover_z_m": self.config["hover_z_m"],
                                   "duration_s": sum(p.duration + 0.75 for p in paths)}
                except ValueError as exc:
                    reasons.append(str(exc))
        raise ValueError(f"No screened approach/transfer path for {color}: {reasons[-4:]}")

    def grid(self):
        rows = []
        table_z = self.config["table_center_m"][2] + self.config["table_half_size_m"][2]
        for x in self.config["grid_x_m"]:
            for y in self.config["grid_y_m"]:
                passed, failure, chosen_yaw = False, "no solution", None
                for yaw in self.config["yaw_candidates_deg"]:
                    quaternion = downward_quaternion(np.deg2rad(yaw))
                    for seed in self.seeds([x, y, self.config["hover_z_m"]]):
                        hover = self.ik.solve([x, y, self.config["hover_z_m"]], quaternion, seed)
                        if not hover.success:
                            failure = hover.reason
                            continue
                        try:
                            self.joint_path("grid approach", self.robot.home, hover.q, quaternion,
                                            [x, y, self.config["hover_z_m"]])
                            self.cartesian_path("grid descent", hover.q, [x, y, table_z + 0.02], quaternion)
                            passed, chosen_yaw = True, yaw
                            break
                        except ValueError as exc:
                            failure = str(exc)
                    if passed:
                        break
                rows.append({"x_m": x, "y_m": y, "passed": passed, "yaw_deg": chosen_yaw,
                             "reason": "ok" if passed else failure})
        return rows
