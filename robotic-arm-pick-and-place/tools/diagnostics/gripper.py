"""Step 2: geometric calibration and physical finger tests in a supported fixture.

Six extra arm-lock constraints and a cube pedestal exist only in this in-memory
diagnostic model. The task scene stays unchanged. This is not an arm-holding or
lifting test. No finger pose is overwritten while physics is advancing.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from robots.forte_gripper import ForteGripper
sys.path.insert(0, str(PROJECT / "tools"))
from build_forte_scene import ARM_JOINTS, HOME, ROBOT, SCENE, digest, require

CUBE_SIZE = 0.04
DEFAULT_REPORT = PROJECT / "outputs" / "diagnostics" / "forte" / "gripper_calibration.json"


def geometry_checks() -> dict:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    grip = ForteGripper(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    site = model.site(grip.config["tool_site"])
    require(int(site.bodyid[0]) == model.body(grip.config["tool_body"]).id, "Wrong tool body")
    np.testing.assert_allclose(site.pos, grip.config["tool_offset_m"], atol=1e-12)
    np.testing.assert_allclose(site.quat, [1, 0, 0, 0], atol=1e-12)
    samples = []
    for width in (grip.max_width, 0.069, CUBE_SIZE, 0.030, grip.min_width):
        grip.initialize(data, width)
        mujoco.mj_forward(model, data)
        measured, center_error = grip.geometry(data)
        require(abs(measured - width) < 1e-9, "Opening formula differs from collision geometry")
        require(center_error < 1e-9, "Grasp site is not centered between fingers")
        require(not any(c.dist < -1e-5 for c in data.contact), "Finger sweep creates penetration")
        samples.append({"driver_m": float(data.qpos[grip.driver_qpos]), "opening_m": measured})

    # Support extent along local +X for every palm/finger collision primitive.
    # This estimates table clearance for a centered cube and +X-down approach.
    rotation = data.site_xmat[grip.site].reshape(3, 3)
    origin = data.site_xpos[grip.site]
    bodies = {model.body(n).id for n in ("wrist_link", "l_end_effector_link", "r_end_effector_link")}
    furthest = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies or model.geom_contype[g] == 0:
            continue
        direction = rotation[:, 0] @ data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            extent = float(np.abs(direction) @ model.geom_size[g])
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = float(model.geom_size[g, 0] * np.linalg.norm(direction[:2])
                           + model.geom_size[g, 1] * abs(direction[2]))
        else:
            raise ValueError("Unexpected gripper collision primitive in clearance calculation")
        furthest = max(furthest, float(rotation[:, 0] @ (data.geom_xpos[g] - origin)) + extent)
    clearance = CUBE_SIZE / 2 - furthest
    require(clearance > 0.005, "Insufficient nominal finger/table clearance")
    # Unsupported inputs must fail rather than silently commanding excess travel.
    for width in (-0.01, grip.max_width + 0.001, float("nan")):
        try:
            grip.position_from_width(width)
        except ValueError:
            pass
        else:
            raise ValueError("Invalid gripper opening was accepted")
    # Exercise saturation deliberately without stepping an abrupt command.
    grip.initialize(data, grip.max_width)
    data.qfrc_applied[:] = 0
    effort = grip.apply(data, grip.min_width)
    require(effort == grip.config["effort_limit_n"], "Closing force cap is not enforced")
    require(np.count_nonzero(data.qfrc_applied) == 1
            and data.qfrc_applied[grip.follower_dof] == 0,
            "Gripper force leaked into the arm, follower, or cube coordinates")
    grip.initialize(data, grip.min_width)
    require(grip.apply(data, grip.max_width) == -grip.config["effort_limit_n"],
            "Opening force cap is not enforced")
    return {"opening_samples": samples, "tool_offset_m": grip.config["tool_offset_m"],
            "approach_axis_local": grip.config["approach_axis_local"],
            "nominal_table_clearance_m": clearance}


def fixture() -> tuple[mujoco.MjModel, mujoco.MjData, ForteGripper, np.ndarray]:
    base = mujoco.MjModel.from_xml_path(str(SCENE))
    initial = mujoco.MjData(base)
    mujoco.mj_resetDataKeyframe(base, initial, base.key("home").id)
    mujoco.mj_forward(base, initial)
    tool_position = initial.site("grasp_site").xpos.copy()
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.delete(spec.key("home"))
    for name, position in zip(ARM_JOINTS, HOME):
        lock = spec.add_equality(name=f"calibration_lock_{name}",
                                 type=mujoco.mjtEq.mjEQ_JOINT, name1=name)
        lock.data[:5] = (position, 0, 0, 0, 0)
        lock.solref[:] = (0.002, 1)
        lock.solimp[:] = (0.999, 0.999, 0.001, 0.5, 2)
    body = spec.worldbody.add_body(name="calibration_cube", pos=tool_position)
    body.add_freejoint()
    body.add_geom(name="calibration_cube", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=(CUBE_SIZE / 2,) * 3, mass=0.05, rgba=(1, 0.65, 0, 1),
                  group=2, friction=(1, 0.5, 0.01))
    top = tool_position[2] - CUBE_SIZE / 2
    spec.worldbody.add_geom(name="calibration_support", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=(tool_position[0], tool_position[1], (0.2 + top) / 2),
                           size=(0.014, 0.012, (top - 0.2) / 2),
                           rgba=(0.4, 0.4, 0.4, 1), group=2)
    model = spec.compile()
    model.vis.scale.framelength = 0.025
    model.vis.scale.framewidth = 0.001
    return model, mujoco.MjData(model), ForteGripper(model), tool_position


def reset_fixture(model, data, grip, tool_position, with_cube):
    mujoco.mj_resetData(model, data)
    for name, position in zip(ARM_JOINTS, HOME):
        data.qpos[model.joint(name).qposadr[0]] = position
    grip.initialize(data, grip.config["open_width_m"])
    for color in ("red", "green", "blue"):
        j = int(model.body(f"cube_{color}").jntadr[0])
        data.qpos[model.jnt_qposadr[j] + 2] = 0.22
    j = int(model.body("calibration_cube").jntadr[0])
    address = int(model.jnt_qposadr[j])
    point = tool_position if with_cube else (0.55, -0.30, 0.22)
    data.qpos[address:address + 7] = [*point, 1, 0, 0, 0]
    # Move the test pedestal out of the jaws during the empty-close test.
    # This is fixture setup between runs, not a collision mask or state clamp.
    support = model.geom("calibration_support").id
    model.geom_pos[support, 1] = tool_position[1] if with_cube else 0.28
    mujoco.mj_forward(model, data)


def contacts(model, data):
    normals = {"left": 0.0, "right": 0.0}
    cube = model.geom("calibration_cube").id
    robot_bodies = {model.body(n).id for n in (
        "shoulder_yaw_link", "shoulder_pitch_link", "upper_arm_link", "elbow_link",
        "lower_arm_link", "wrist_link", "l_end_effector_link", "r_end_effector_link",
    )}
    deepest = 0.0
    forbidden = []
    for i, contact in enumerate(data.contact):
        a, b = int(contact.geom1), int(contact.geom2)
        name_a, name_b = model.geom(a).name, model.geom(b).name
        deepest = max(deepest, -float(contact.dist))
        if cube in (a, b):
            other = model.geom(b if a == cube else a).name
            for side, prefix in (("left", "l_finger_"), ("right", "r_finger_")):
                if other.startswith(prefix):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    normals[side] += max(0.0, float(force[0]))
        if contact.dist < -0.0005:
            forbidden.append(f"{name_a}/{name_b}: {contact.dist * 1000:.3f} mm")
        # Robot contacts are allowed only at the diagnostic cube's finger faces.
        robot_a = model.geom_bodyid[a] in robot_bodies
        robot_b = model.geom_bodyid[b] in robot_bodies
        if (robot_a or robot_b) and contact.dist < -1e-5:
            other_name = model.geom(b if a == cube else a).name if cube in (a, b) else ""
            allowed = cube in (a, b) and other_name.startswith(("l_finger_finger_", "r_finger_finger_"))
            if not allowed:
                forbidden.append(f"unexpected robot contact: {name_a}/{name_b}")
    return normals, deepest, forbidden


class CalibrationRun:
    def __init__(self, model, data, grip, viewer=None):
        self.model, self.data, self.grip, self.viewer = model, data, grip, viewer
        self.peak_effort = self.peak_speed = self.peak_mimic = self.peak_center = 0.0
        self.peak_penetration = self.peak_arm_drift = 0.0
        self.wall_start = time.perf_counter()
        self.simulated = 0.0
        self.arm_addresses = np.array([model.joint(n).qposadr[0] for n in ARM_JOINTS])

    def segment(self, start_width, end_width, seconds, verify_contact=False, verify_release=False):
        count = round(seconds / self.model.opt.timestep)
        for index in range(count):
            if self.viewer is not None and not self.viewer.is_running():
                raise InterruptedError("Calibration viewer closed before the run completed")
            fraction = (index + 1) / count
            blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
            derivative = (30 * fraction**2 - 60 * fraction**3 + 30 * fraction**4) / seconds
            width = start_width + blend * (end_width - start_width)
            driver_speed = -0.5 * derivative * (end_width - start_width)
            self.data.qfrc_applied[:] = 0
            effort = self.grip.apply(self.data, width, driver_speed)
            mujoco.mj_step(self.model, self.data)
            # Refresh contacts/forces at the integrated state for reporting.
            mujoco.mj_forward(self.model, self.data)
            require(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all(),
                    "Nonfinite state during gripper calibration")
            require(not any(w.number for w in self.data.warning), "MuJoCo numerical warning")
            self.peak_effort = max(self.peak_effort, abs(effort))
            speed = float(np.max(np.abs(self.data.qvel[[self.grip.driver_dof, self.grip.follower_dof]])))
            self.peak_speed = max(self.peak_speed, speed)
            mimic = abs(float(self.data.qpos[self.grip.driver_qpos] + self.data.qpos[self.grip.follower_qpos]))
            self.peak_mimic = max(self.peak_mimic, mimic)
            _, center_error = self.grip.geometry(self.data)
            self.peak_center = max(self.peak_center, center_error)
            arm_drift = float(np.max(np.abs(self.data.qpos[self.arm_addresses] - HOME)))
            self.peak_arm_drift = max(self.peak_arm_drift, arm_drift)
            normals, penetration, forbidden = contacts(self.model, self.data)
            self.peak_penetration = max(self.peak_penetration, penetration)
            require(not forbidden, "; ".join(forbidden))
            if index >= count - round(0.2 / self.model.opt.timestep):
                if verify_contact:
                    require(all(value > 0.2 for value in normals.values()),
                            "Bilateral contact was not sustained for the final 0.2 s")
                if verify_release:
                    require(sum(normals.values()) < 0.01, "Finger contact persisted after release")
            require(speed <= self.grip.config["speed_limit_m_per_s"] + 0.001,
                    "Measured slider speed exceeds 0.1 m/s")
            require(mimic < 0.0001 and center_error < 0.0001, "Finger coupling/centering error")
            require(arm_drift < 0.0001, "Calibration arm fixture moved excessively")
            for address, joint_name in ((self.grip.driver_qpos, "driver_joint"),
                                        (self.grip.follower_qpos, "follower_joint")):
                lo, hi = self.model.joint(self.grip.config[joint_name]).range
                require(lo - 0.0001 <= self.data.qpos[address] <= hi + 0.0001,
                        "Measured slider travel exceeded its limit")
            self.simulated += self.model.opt.timestep
            if self.viewer is not None and index % 16 == 0:
                self.viewer.sync()
                time.sleep(max(0, self.wall_start + self.simulated - time.perf_counter()))


def save_snapshot(model, data, path):
    import imageio.v2 as imageio

    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.site("grasp_site").xpos
    camera.distance, camera.azimuth, camera.elevation = 0.38, 135, -25
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    option.frame = mujoco.mjtFrame.mjFRAME_SITE
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        imageio.imwrite(path, renderer.render())


def run(args):
    geometry = geometry_checks()
    print("PASS: opening geometry, transverse grasp center, slider limits, force caps, and invalid inputs")
    print(f"Nominal +X-down clearance above a flat table: {geometry['nominal_table_clearance_m'] * 1000:.2f} mm")
    model, data, grip, tool_position = fixture()
    reset_fixture(model, data, grip, tool_position, False)
    context = nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(model, data)
    contact_results = []
    with context as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.lookat[:] = tool_position
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.45, 135, -25
                viewer.opt.geomgroup[0] = int(args.collision_geoms)
                viewer.opt.geomgroup[1] = int(not args.collision_geoms)
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        session = CalibrationRun(model, data, grip, viewer)
        opening = grip.config["open_width_m"]
        closing = grip.config["empty_close_width_m"]
        gripping = grip.config["grasp_reference_width_m"]
        for cycle in range(args.cycles):
            print(f"Cycle {cycle + 1}: EMPTY open -> close -> open (arm held by fixture)")
            reset_fixture(model, data, grip, tool_position, False)
            session.segment(opening, opening, 0.5)
            session.segment(opening, closing, 2.0)
            session.segment(closing, closing, 0.75)
            gap, _ = grip.geometry(data)
            require(abs(gap - closing) < 0.0002, "Empty gripper did not reach requested close width")
            session.segment(closing, opening, 2.0)
            session.segment(opening, opening, 0.75)
            require(abs(grip.geometry(data)[0] - opening) < 0.0002, "Empty gripper did not reopen")

            print("CONTACT: fixture reset places an orange 4 cm cube on the pedestal")
            reset_fixture(model, data, grip, tool_position, True)
            session.segment(opening, opening, 0.5)
            session.segment(opening, gripping, 2.0)
            session.segment(gripping, gripping, 1.0, verify_contact=True)
            normals, _, _ = contacts(model, data)
            gap, center_error = grip.geometry(data)
            require(all(force > 0.2 for force in normals.values()), "No sustained bilateral finger contact")
            require(abs(gap - CUBE_SIZE) < 0.0005, "Contact opening differs from cube width")
            require(abs(gap - gripping) > 0.005, "Fingers passed through the cube toward the reference")
            cube_drift = float(np.linalg.norm(data.body("calibration_cube").xpos - tool_position))
            require(cube_drift < 0.001, "Supported cube moved excessively during closure")
            result = {"measured_gap_m": gap, "normal_forces_n": normals,
                      "center_error_m": center_error, "cube_drift_m": cube_drift,
                      "bilateral_hold_verified_s": 0.2}
            contact_results.append(result)
            print(f"PASS: contact gap {gap * 1000:.3f} mm; L/R normal forces "
                  f"{normals['left']:.3f}/{normals['right']:.3f} N")
            if args.snapshot and cycle == args.cycles - 1:
                save_snapshot(model, data, args.snapshot)
            session.segment(gripping, opening, 2.0)
            session.segment(opening, opening, 0.75, verify_release=True)
            normals, _, _ = contacts(model, data)
            require(sum(normals.values()) < 0.01, "Cube was not released")
        metrics = {
            "peak_driver_effort_n": session.peak_effort,
            "peak_slider_speed_m_per_s": session.peak_speed,
            "peak_coupling_error_m": session.peak_mimic,
            "peak_transverse_center_error_m": session.peak_center,
            "peak_penetration_m": session.peak_penetration,
            "peak_arm_fixture_drift_rad": session.peak_arm_drift,
            "mujoco_warning_count": sum(int(w.number) for w in data.warning),
        }
    report = {"status": "PASS", "mujoco_version": mujoco.__version__,
              "scope": "supported gripper fixture; no arm PD, lifting, or IK validation",
              "configuration": grip.config,
              "robot_sha256": digest(ROBOT), "scene_sha256": digest(SCENE),
              "cycles": args.cycles, "geometry": geometry,
              "physical_metrics": metrics, "contact_results": contact_results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: force/speed/travel bounds, physical mimic coupling, contact and release; {args.cycles} cycle(s)")
    print(f"Peak force {session.peak_effort:.3f} N; speed {session.peak_speed:.4f} m/s; "
          f"penetration {session.peak_penetration * 1000:.3f} mm")
    print(f"Report: {args.report.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--viewer", action="store_true", help="Watch calibration at real-time speed")
    mode.add_argument("--headless", action="store_true", help="Run physical checks without a window (default)")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--collision-geoms", action="store_true", help="Inspect primitives in the viewer")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--snapshot", type=Path, help="Save the final contact pose as a PNG")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if args.collision_geoms and not args.viewer:
        parser.error("--collision-geoms requires --viewer")
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError, InterruptedError) as exc:
        parser.exit(1, f"Calibration incomplete/failed: {exc}\n")


if __name__ == "__main__":
    main()
