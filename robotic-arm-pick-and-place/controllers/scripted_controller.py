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
