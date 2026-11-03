"""
controllers/scripted_controller.py  —  Phase 3 rule-based pick & place.

Provides ScriptedController: a class that drives the Panda arm to pick a cube
and place it on its colored target pad, with success/failure detection.

Key idea: we use INVERSE KINEMATICS (IK) to turn a desired gripper position
(x, y, z in the world) into the 7 joint angles that put the gripper there.
MuJoCo gives us the Jacobian, and we solve it with damped least squares.
"""

import time
import numpy as np
import mujoco


class ScriptedController:
    # The arm's resting pose (7 joint angles, radians).
    HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    GRIPPER_OPEN = 255.0
    GRIPPER_CLOSE = 0.0

    # Heights (z, metres) used during the routine.
    HOVER_Z = 0.40     # safe travel height above the table
    GRASP_Z = 0.225    # gripper height to grab a cube (cube centre ~0.22)
    PLACE_Z = 0.245    # gripper height to release onto a target pad
    SUCCESS_TOL = 0.05  # cube must land within 5 cm of the target (xy)

    def __init__(self, model, data, seed=0):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.jnt_range = model.jnt_range[:7].copy()  # arm joint limits
        self.recorder = None          # optional: set to a Recorder to film the run
        self.rng = np.random.default_rng(seed)
        # Capture a "gripper pointing straight down" orientation once, by
        # posing the arm in a known downward configuration and reading it.
        data.qpos[:7] = [0, 0.3, 0, -2.2, 0, 2.5, 0.785]
        mujoco.mj_forward(model, data)
        self.down_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.down_quat, data.site_xmat[self.site_id])

    # ---- low-level helpers ------------------------------------------------
    def reset(self, randomize=False):
        """Cubes back on the table, arm at home, gripper open.
        If randomize=True, scatter the cubes to random reachable spots."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.HOME
        self.data.ctrl[:7] = self.HOME
        self.data.ctrl[7] = self.GRIPPER_OPEN
        if randomize:
            for name in ("cube_red", "cube_green", "cube_blue"):
                jid = self.model.body(name).jntadr[0]
                adr = self.model.jnt_qposadr[jid]
                x = self.rng.uniform(0.38, 0.55)
                y = self.rng.uniform(-0.22, 0.22)
                self.data.qpos[adr:adr + 7] = [x, y, 0.25, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def _capture(self):
        if self.recorder is not None:
            self.recorder.capture(self.data)

    def ik(self, target_pos, q_seed):
        """
        Solve for the 7 arm angles that place the gripper at target_pos
        (pointing down). Runs on a scratch copy so the live sim is untouched.
        Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * error.
        """
        sd = mujoco.MjData(self.model)
        sd.qpos[:] = self.data.qpos[:]
        sd.qpos[:7] = q_seed
        for _ in range(200):
            mujoco.mj_forward(self.model, sd)
            err_pos = np.asarray(target_pos) - sd.site_xpos[self.site_id]
            cq = np.zeros(4); mujoco.mju_mat2Quat(cq, sd.site_xmat[self.site_id])
            nq = np.zeros(4); mujoco.mju_negQuat(nq, cq)
            eq = np.zeros(4); mujoco.mju_mulQuat(eq, self.down_quat, nq)
            err_rot = np.zeros(3); mujoco.mju_quat2Vel(err_rot, eq, 1.0)
            err = np.concatenate([err_pos, err_rot])
            if np.linalg.norm(err) < 1e-3:
                break
            jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, sd, jp, jr, self.site_id)
            J = np.vstack([jp[:, :7], jr[:, :7]])           # 6x7 Jacobian
            dq = J.T @ np.linalg.solve(J @ J.T + 0.04 * np.eye(6), err)
            sd.qpos[:7] = np.clip(sd.qpos[:7] + dq,
                                  self.jnt_range[:, 0], self.jnt_range[:, 1])
        return sd.qpos[:7].copy()

    def _move_to(self, arm_q, grip, seconds, viewer=None):
        """Smoothly ramp the servo targets to arm_q over `seconds`."""
        start = self.data.ctrl[:7].copy()
        n = int(seconds / self.model.opt.timestep)
        for i in range(n):
            if viewer is not None and not viewer.is_running():
                return
            alpha = (i + 1) / n
            self.data.ctrl[:7] = start + alpha * (arm_q - start)
            self.data.ctrl[7] = grip
            mujoco.mj_step(self.model, self.data)
            self._capture()
            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep)

    def _settle(self, steps, viewer=None):
        for _ in range(steps):
            if viewer is not None and not viewer.is_running():
                return
            mujoco.mj_step(self.model, self.data)
            self._capture()
            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep)

    def _body_pos(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def _site_pos(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[sid].copy()

    # ---- the full pick-and-place routine ----------------------------------
    def pick_and_place(self, cube_body, target_site, viewer=None):
        """
        Pick up `cube_body` and place it on `target_site`.
        Returns True if the cube ends up within SUCCESS_TOL of the target.
        """
        cube = self._body_pos(cube_body)
        cx, cy = cube[0], cube[1]

        # 1. hover above the cube, gripper open
        q = self.ik([cx, cy, self.HOVER_Z], self.HOME)
        self._move_to(q, self.GRIPPER_OPEN, 1.5, viewer)
        # 2. descend to grasp height
        q = self.ik([cx, cy, self.GRASP_Z], q)
        self._move_to(q, self.GRIPPER_OPEN, 1.5, viewer)
        # 3. close the gripper and let the grip settle
        self._move_to(q, self.GRIPPER_CLOSE, 1.0, viewer)
        self._settle(200, viewer)
        # 4. lift straight up
        q = self.ik([cx, cy, self.HOVER_Z], q)
        self._move_to(q, self.GRIPPER_CLOSE, 1.5, viewer)

        # 5. move over the target, descend, release
        tgt = self._site_pos(target_site)
        q = self.ik([tgt[0], tgt[1], self.HOVER_Z], q)
        self._move_to(q, self.GRIPPER_CLOSE, 1.5, viewer)
        q = self.ik([tgt[0], tgt[1], self.PLACE_Z], q)
        self._move_to(q, self.GRIPPER_CLOSE, 1.5, viewer)
        self._move_to(q, self.GRIPPER_OPEN, 1.0, viewer)
        self._settle(200, viewer)

        # 6. retreat back to home so we don't knock other cubes
        q = self.ik([tgt[0], tgt[1], self.HOVER_Z], q)
        self._move_to(q, self.GRIPPER_OPEN, 1.0, viewer)
        self._move_to(self.HOME, self.GRIPPER_OPEN, 1.5, viewer)

        # 7. success check: did the cube land near the target pad?
        final = self._body_pos(cube_body)
        dist = np.linalg.norm(final[:2] - tgt[:2])
        on_table = final[2] > 0.18           # didn't fall off the table
        return bool(dist < self.SUCCESS_TOL and on_table)


# The original Panda controller above remains available unchanged. Both robots
# share these entry-point task names and the controller factory below.
TASKS = [(f"cube_{color}", f"target_{color}") for color in ("green", "red", "blue")]


class ForteController:
    """Physical contact grasping with screened paths and continuous arm control."""

    def __init__(self, robot, seed=0):
        from .reaching import ReachPlanner
        self.robot, self.model, self.data = robot, robot.model, robot.data
        self.rng = np.random.default_rng(seed)
        self.recorder = None
        self.verbose = True
        self.planner = ReachPlanner(robot)
        self.last_result = None
        self.results = []
        self.active_cube = None
        self.contact_allowed = False
        self._counter = 0
        self._viewer = None
        self._wall_start = None
        self._peak_torque = 0.0
        self._saturation = 0
        self._peak_slip = 0.0
        self._peak_gripper_effort = 0.0
        self._peak_penetration = 0.0
        self._grasp_offset = None
        self._names = [self.model.geom(i).name for i in range(self.model.ngeom)]

    def reset(self, randomize=False):
        self.robot.reset()
        self.planner.ik.set_payload(None)
        self.active_cube, self.contact_allowed = None, False
        self._grasp_offset = None
        self.results, self.last_result = [], None
        self._viewer, self._wall_start = None, None
        if randomize:
            cfg = self.planner.config
            for color, initial in cfg["cube_spawn_m"].items():
                joint = int(self.model.body(f"cube_{color}").jntadr[0])
                address = self.model.jnt_qposadr[joint]
                for axis, bounds in ((0, cfg["grid_x_m"]), (1, cfg["grid_y_m"])):
                    lo, hi = max(min(bounds), initial[axis] - 0.008), min(max(bounds), initial[axis] + 0.008)
                    self.data.qpos[address + axis] = self.rng.uniform(lo, hi)
            mujoco.mj_forward(self.model, self.data)

    def _say(self, message):
        if self.verbose:
            print(message, flush=True)

    def _step(self, viewer=None):
        if viewer is not None and not viewer.is_running():
            raise InterruptedError("Viewer closed")
        self.robot.step()
        self._peak_torque = max(self._peak_torque, float(np.max(np.abs(self.robot.last_effort))))
        self._saturation += int(np.any(np.abs(self.robot.last_raw_effort) > self.robot.effort_limits + 1e-8))
        self._peak_gripper_effort = max(self._peak_gripper_effort, abs(self.robot.last_gripper_effort))
        if self._grasp_offset is not None:
            rotation = self.data.site_xmat[self.robot.site_id].reshape(3, 3)
            measured = rotation.T @ (self.data.body(self.active_cube).xpos - self.robot.tool_position)
            slip = float(np.linalg.norm(measured - self._grasp_offset))
            self._peak_slip = max(self._peak_slip, slip)
            if slip > 0.003:
                raise RuntimeError("Cube slipped more than 3 mm within the grasp")
        if np.any(self.robot.arm_position < self.robot.lower - 0.001) or np.any(self.robot.arm_position > self.robot.upper + 0.001):
            raise RuntimeError("Arm joint travel exceeded")
        if np.any(np.abs(self.robot.arm_velocity) > self.robot.speed_limits):
            raise RuntimeError("Arm speed exceeded")
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if not (self.planner.ik.robot_geoms[a] or self.planner.ik.robot_geoms[b]):
                continue
            self._peak_penetration = max(self._peak_penetration, -float(contact.dist))
            allowed = False
            if self.contact_allowed and self.active_cube is not None:
                held = self.model.geom(self.active_cube).id
                if held in (a, b):
                    other = b if a == held else a
                    allowed = self._names[other].startswith(("l_finger_finger_", "r_finger_finger_"))
            if contact.dist < (-0.0005 if allowed else -1e-5):
                raise RuntimeError(f"Unexpected contact/penetration: {self._names[a]}/{self._names[b]}")
        if self.recorder is not None:
            self.recorder.capture(self.data)
        self._counter += 1
        if viewer is not None and self._counter % 16 == 0:
            if viewer is not self._viewer:
                self._viewer = viewer
                self._wall_start = time.perf_counter() - self.data.time
            viewer.sync()
            time.sleep(max(0, self._wall_start + self.data.time - time.perf_counter()))

    def _settle(self, steps, viewer=None):
        for _ in range(steps):
            self._step(viewer)

    def _execute(self, path, width, viewer=None):
        from .ik import rotation_error
        self._say(path.label)
        steps = int(np.ceil(path.duration / self.model.opt.timestep))
        for i in range(steps):
            q, dq = path.sample((i + 1) * self.model.opt.timestep)
            self.robot.set_arm_target(q, dq)
            self.robot.set_gripper_width(width)
            self._step(viewer)
        self.robot.set_arm_target(path.nodes[-1])
        self._settle(round(0.75 / self.model.opt.timestep), viewer)
        position_error = np.linalg.norm(self.robot.tool_position - path.desired_position)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.robot.site_id])
        angle_error = float(np.linalg.norm(rotation_error(path.desired_quaternion, quat)))
        if position_error > 0.002 or angle_error > np.deg2rad(2):
            raise RuntimeError(f"Pose tracking timeout at {path.label}: {position_error * 1000:.3f} mm, {np.degrees(angle_error):.3f} deg")

    def _grip(self, start, end, viewer=None):
        seconds = 2.0
        steps = round(seconds / self.model.opt.timestep)
        for i in range(steps):
            alpha = (i + 1) / steps
            blend = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
            derivative = (30 * alpha**2 - 60 * alpha**3 + 30 * alpha**4) / seconds
            self.robot.set_gripper_width(start + blend * (end - start), -0.5 * derivative * (end - start))
            self._step(viewer)
        self.robot.set_gripper_width(end)
        self._settle(round(0.75 / self.model.opt.timestep), viewer)

    def _normal_forces(self, body):
        cube = self.model.geom(body).id
        forces = np.zeros(2)
        force = np.zeros(6)
        for i, c in enumerate(self.data.contact):
            a, b = int(c.geom1), int(c.geom2)
            if cube not in (a, b):
                continue
            name = self._names[b if a == cube else a]
            for side, prefix in enumerate(("l_finger_finger_", "r_finger_finger_")):
                if name.startswith(prefix):
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    forces[side] += max(0, float(force[0]))
        return forces

    def pick_and_place(self, cube_body, target_site, viewer=None):
        color = cube_body.removeprefix("cube_")
        if (cube_body, target_site) not in TASKS:
            raise ValueError("Select a named cube and its matching target")
        self._peak_torque, self._saturation = 0.0, 0
        self._peak_slip, self._peak_gripper_effort, self._peak_penetration = 0.0, 0.0, 0.0
        result = {"cube": cube_body, "target": target_site, "success": False}
        try:
            paths, info = self.planner.plan_task(color)
            self.active_cube = cube_body
            opened = self.robot.open_width
            gripping = self.robot.gripper.config["grasp_reference_width_m"]
            for path in paths[:2]:
                self._execute(path, opened, viewer)
            rest_z = float(self.data.body(cube_body).xpos[2])
            self._say(f"{color}: close and verify bilateral contact")
            self.contact_allowed = True
            self._grip(opened, gripping, viewer)
            forces = self._normal_forces(cube_body)
            if np.any(forces < 0.2):
                raise RuntimeError("No bilateral grasp contact")
            result["grasp_normal_forces_n"] = forces.tolist()
            self.robot.set_payload(cube_body)
            self.planner.ik.set_payload(cube_body)
            self._grasp_offset = self.planner.ik.payload["offset"].copy()
            from .ik import downward_quaternion
            quat = downward_quaternion(np.deg2rad(info["yaw_deg"]))
            cube = self.data.body(cube_body).xpos.copy()
            hover = np.array([cube[0], cube[1], self.planner.config["hover_z_m"]])
            lift = self.planner.cartesian_path(f"{color}: lift cube", self.robot.arm_position, hover, quat)
            self._execute(lift, gripping, viewer)
            lifted = float(self.data.body(cube_body).xpos[2] - rest_z)
            result["lift_height_m"] = lifted
            if lifted < 0.035 or np.any(self._normal_forces(cube_body) < 0.2):
                raise RuntimeError("Cube was not lifted securely")
            target = self.data.site(target_site).xpos.copy()
            over = np.array([target[0], target[1], self.planner.config["hover_z_m"]])
            travel = self.planner.cartesian_path(f"{color}: carry cube to target", self.robot.arm_position, over, quat)
            self._execute(travel, gripping, viewer)
            # Refresh the measured grasp transform before planning release.
            self.planner.ik.set_payload(cube_body)
            offset = self.planner.ik.payload["offset"]
            rotation = self.data.site_xmat[self.robot.site_id].reshape(3, 3)
            table_z = self.planner.config["table_center_m"][2] + self.planner.config["table_half_size_m"][2]
            half = float(self.model.geom(cube_body).size[2])
            release_cube = np.array([target[0], target[1], table_z + half + self.planner.config["release_clearance_m"]])
            release_tool = release_cube - rotation @ offset
            lower = self.planner.cartesian_path(f"{color}: lower cube", self.robot.arm_position, release_tool, quat)
            self._execute(lower, gripping, viewer)
            if np.any(self._normal_forces(cube_body) < 0.2):
                raise RuntimeError("Grasp lost during transfer")
            self._say(f"{color}: release cube")
            self._grasp_offset = None
            self._grip(gripping, opened, viewer)
            self.robot.set_payload(None)
            self.planner.ik.set_payload(None)
            self.contact_allowed = False
            retreat = self.planner.cartesian_path(f"{color}: retreat", self.robot.arm_position, over, quat)
            self._execute(retreat, opened, viewer)
            home_p, home_q = self.planner.ik.pose(self.robot.home)
            home = self.planner.joint_path(f"{color}: return home", self.robot.arm_position,
                                           self.robot.home, home_q, home_p)
            self._execute(home, opened, viewer)
            final = self.data.body(cube_body).xpos.copy()
            joint = int(self.model.body(cube_body).jntadr[0])
            velocity = self.data.qvel[self.model.jnt_dofadr[joint]:self.model.jnt_dofadr[joint] + 6]
            distance = float(np.linalg.norm(final[:2] - target[:2]))
            released = bool(np.sum(self._normal_forces(cube_body)) < 0.01)
            stable = bool(np.linalg.norm(velocity[:3]) < 0.01 and np.linalg.norm(velocity[3:]) < 0.1)
            success = bool(distance < 0.02 and abs(final[2] - table_z - half) < 0.003 and released and stable)
            result.update({"success": success, "placement_error_m": distance, "final_position_m": final.tolist(),
                           "released": released, "stable": stable})
            if not success:
                result["reason"] = "Final released placement did not meet position/stability tolerances"
        except (ValueError, RuntimeError, InterruptedError) as exc:
            result["reason"] = str(exc)
            result["failure_tool_m"] = self.robot.tool_position.tolist()
            result["failure_cube_m"] = self.data.body(cube_body).xpos.tolist()
        result.update({"peak_arm_effort_nm": self._peak_torque, "saturated_steps": self._saturation,
                       "peak_grasp_slip_m": self._peak_slip,
                       "peak_gripper_driver_effort_n": self._peak_gripper_effort,
                       "peak_robot_contact_penetration_m": self._peak_penetration,
                       "mujoco_warning_count": sum(int(w.number) for w in self.data.warning)})
        self.last_result = result
        self.results.append(result)
        if not result["success"]:
            self._say(f"FAILED: {result.get('reason')}")
        return result["success"]


def make_controller(robot, seed=0):
    from robots.forte import ForteRobot
    return ForteController(robot, seed) if isinstance(robot, ForteRobot) else ScriptedController(robot.model, robot.data, seed)
