"""IK contracts that matter when exchanging robot/scene models."""

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from controllers.ik import PoseIK, downward_quaternion
from robots.factory import load_robot, SCENES
from robots.forte import ForteRobot


class ForteIKTests(unittest.TestCase):
    def setUp(self):
        self.robot = load_robot("forte")
        self.ik = PoseIK(self.robot)

    def test_full_pose_convergence_preserves_live_state(self):
        data = self.robot.data
        state = {name: getattr(data, name).copy() for name in
                 ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")}
        result = self.ik.solve([0.45, 0, 0.27], downward_quaternion(), self.robot.home)
        self.assertTrue(result.success, result.reason)
        self.assertLessEqual(result.position_error_m, 0.0005)
        self.assertLessEqual(result.rotation_error_rad, np.deg2rad(0.5))
        for name, value in state.items():
            np.testing.assert_array_equal(getattr(data, name), value)
        self.assertEqual(data.time, 0)
        opposite = self.ik.solve([0.45, 0, 0.27], -downward_quaternion(), self.robot.home)
        self.assertTrue(opposite.success, opposite.reason)
        np.testing.assert_allclose(result.q, opposite.q, atol=1e-9)

    def test_unreachable_and_invalid_requests_fail(self):
        result = self.ik.solve([3, 0, 1], downward_quaternion(), self.robot.home, max_iterations=100)
        self.assertFalse(result.success)
        for position, quaternion in (([np.nan, 0, 0], [1, 0, 0, 0]),
                                      ([0.5, 0, 0.27], [0, 0, 0, 0])):
            with self.assertRaises(ValueError):
                self.ik.solve(position, quaternion, self.robot.home)

    def test_payload_planning_preserves_live_objects_and_applies_only_robot_forces(self):
        robot = self.robot
        before = robot.data.qpos.copy()
        self.ik.set_payload("cube_green")
        offset = self.ik.payload["offset"].copy()
        candidate = robot.home.copy()
        candidate[0] += 0.1
        position, _ = self.ik.pose(candidate)
        rotation = self.ik.data.site_xmat[robot.site_id].reshape(3, 3)
        np.testing.assert_allclose(self.ik.data.body("cube_green").xpos, position + rotation @ offset)
        np.testing.assert_array_equal(robot.data.qpos, before)
        cube = robot.model.geom("cube_green").id
        self.assertTrue(self.ik.allowed_payload_contact(cube, robot.model.geom("l_finger_finger_taper").id))
        self.assertFalse(self.ik.allowed_payload_contact(cube, robot.model.geom("l_finger_slider_plate").id))
        unloaded = robot.gravity_effort()
        robot.set_payload("cube_green")
        self.assertGreater(np.linalg.norm(robot.gravity_effort() - unloaded), 0.01)
        robot._apply_control()
        controlled = np.r_[robot.dof_indices, robot.gripper.driver_dof]
        uncontrolled = np.setdiff1d(np.arange(robot.model.nv), controlled)
        np.testing.assert_array_equal(robot.data.qfrc_applied[uncontrolled], 0)
        np.testing.assert_array_equal(robot.data.qpos, before)
        self.assertTrue(np.all(np.abs(robot.last_effort) <= robot.effort_limits))
        self.assertEqual(robot.model.neq, 1)  # Only the source finger mimic; no grasp weld.

    def test_solved_but_obstructed_pose_is_rejected(self):
        spec = mujoco.MjSpec.from_file(str(SCENES["forte"]))
        spec.worldbody.add_geom(name="test_obstacle", type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=self.robot.tool_position, size=(0.1, 0.1, 0.1))
        model = spec.compile()
        robot = ForteRobot(model, mujoco.MjData(model))
        robot.reset()
        ik = PoseIK(robot)
        position, quaternion = ik.pose(robot.home)
        result = ik.solve(position, quaternion, robot.home)
        self.assertFalse(result.success)
        self.assertIn("collision", result.reason)

    def test_geometric_solution_with_excess_gravity_torque_is_rejected(self):
        q = np.array([0, -0.35, 0, 0.01, 0, 0.01])
        position, quaternion = self.ik.pose(q)
        self.assertGreater(float(np.max(np.abs(self.ik.data.qfrc_bias[self.robot.dof_indices]))), 10)
        result = self.ik.solve(position, quaternion, q)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "static gravity torque margin")

    def test_joint_mapping_with_free_objects_before_robot(self):
        # Reorder XML bodies so array-prefix indexing would address cube DOFs.
        spec = mujoco.MjSpec.from_file(str(SCENES["forte"]))
        spec.compile()
        root = ET.fromstring(spec.to_xml())
        for mesh in root.findall("asset/mesh"):
            mesh.set("file", str(SCENES["forte"].parent / "source" / Path(mesh.get("file")).name))
        root.find("compiler").attrib.pop("meshdir", None)
        keys = root.find("keyframe")
        if keys is not None:
            root.remove(keys)
        world = root.find("worldbody")
        for color in ("red", "green", "blue"):
            body = world.find(f"body[@name='cube_{color}']")
            world.remove(body)
            world.insert(0, body)
        xml = ET.tostring(root, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml)
        qpos = model.qpos0.copy()
        for name, q in zip(self.robot.joint_names, self.robot.home):
            qpos[model.joint(name).qposadr[0]] = q
        for color in ("red", "green", "blue"):
            j = model.body(f"cube_{color}").jntadr[0]
            qpos[model.jnt_qposadr[j] + 2] = 0.22
        ET.SubElement(ET.SubElement(root, "keyframe"), "key", {
            "name": "home", "qpos": " ".join(map(str, qpos)),
        })
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        robot = ForteRobot(model, mujoco.MjData(model))
        robot.reset()
        self.assertGreaterEqual(int(robot.qpos_indices[0]), 21)
        self.assertGreaterEqual(int(robot.dof_indices[0]), 18)
        result = PoseIK(robot).solve([0.45, 0, 0.27], downward_quaternion(), robot.home)
        self.assertTrue(result.success, result.reason)


if __name__ == "__main__":
    unittest.main()
