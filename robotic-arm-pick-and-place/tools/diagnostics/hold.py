"""Step 3: unbraced gravity hold, disturbance recovery, and gripper operation.

The 14-second test uses the actual task scene, with no added arm locks or
support pedestal. It holds home, briefly pushes the wrist down, recovers, then
opens/closes the empty gripper while maintaining the arm target. No IK yet.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from robots.factory import SCENES, load_robot
from robots.forte import ForteRobot


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(robot):
    """Check bounds, force routing, and gravity evaluation before simulating."""
    robot.reset()
    for invalid in (np.zeros(len(robot.home) + 1), np.full_like(robot.home, np.nan), robot.upper + 0.1):
        try:
            robot.set_arm_target(invalid)
        except ValueError:
            pass
        else:
            raise ValueError("Invalid arm target was accepted")
    try:
        robot.set_arm_target(robot.home, robot.speed_limits + 0.1)
    except ValueError:
        pass
    else:
        raise ValueError("Excessive arm reference speed was accepted")

    if isinstance(robot, ForteRobot):
        require(robot.model.nmocap == 0 and robot.model.neq == 1, "Unexpected fixture/mocap support")
        require(np.all(robot.model.jnt_stiffness[robot.joint_ids] == 0), "Unexpected arm springs")
        expected = robot.gravity_effort()
        robot.data.qvel[robot.dof_indices] = np.linspace(0.1, 0.6, len(robot.home))
        before_q, before_v = robot.data.qpos.copy(), robot.data.qvel.copy()
        np.testing.assert_allclose(robot.gravity_effort(), expected, atol=1e-10)
        np.testing.assert_array_equal(robot.data.qpos, before_q)
        np.testing.assert_array_equal(robot.data.qvel, before_v)
        robot.reset()
        robot.set_arm_target(robot.upper - 0.001)
        robot._apply_control()  # offline saturation test: never step this abrupt reference
        require(np.all(np.abs(robot.last_effort) <= robot.effort_limits), "Torque cap failed")
        require(np.any(np.abs(robot.last_raw_effort) > robot.effort_limits), "Saturation test did not exercise a cap")
        allowed = np.r_[robot.dof_indices, robot.gripper.driver_dof]
        untouched = np.setdiff1d(np.arange(robot.model.nv), allowed)
        require(np.all(robot.data.qfrc_applied[untouched] == 0), "Control force leaked into free objects/follower")
        require(abs(robot.last_gravity[1]) < robot.effort_limits[1], "Home has no gravity holding margin")
        compatibility = None
    else:
        # Prove that the Panda adapter preserves the original position-servo
        # behavior, including its existing gravity sag. Do not retune Panda.
        legacy = mujoco.MjData(robot.model)
        legacy.qpos[:] = robot.data.qpos
        legacy.qvel[:] = robot.data.qvel
        for _ in range(round(2 / robot.model.opt.timestep)):
            legacy.ctrl[:7] = robot.home
            legacy.ctrl[7] = 255
            mujoco.mj_step(robot.model, legacy)
            mujoco.mj_forward(robot.model, legacy)
            robot.step()
        difference = float(np.max(np.abs(robot.data.qpos - legacy.qpos)))
        require(difference < 1e-12, "Panda adapter differs from original direct servo stepping")
        compatibility = {"legacy_state_difference_max": difference}
        settled_tool = robot.tool_position.copy()
        robot.reset()
        compatibility["legacy_home_sag_m"] = float(np.linalg.norm(settled_tool - robot.tool_position))
    robot.reset()
    return compatibility


def reference(t, opened, closed):
    if t < 7:
        return opened, 0.0
    if t < 9:
        fraction, start, end = (t - 7) / 2, opened, closed
    elif t < 10:
        return closed, 0.0
    elif t < 12:
        fraction, start, end = (t - 10) / 2, closed, opened
    else:
        return opened, 0.0
    blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
    derivative = (30 * fraction**2 - 60 * fraction**3 + 30 * fraction**4) / 2
    return start + blend * (end - start), -0.5 * derivative * (end - start)


def save_outputs(path, report, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not rows:
        return
    with gzip.open(path.with_suffix(".csv.gz"), "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(times, [r["tool_error_m"] * 1000 for r in rows])
    axes[0].axhline(report["thresholds"]["settled_tool_error_m"] * 1000,
                   color="gray", linestyle="--", label="Settled tolerance")
    axes[0].set_ylabel("Tool error (mm)")
    axes[0].legend()
    for name in report["joint_names"]:
        axes[1].plot(times, [r[f"{name}_effort_fraction"] for r in rows], label=name)
    axes[1].axhline(1, color="red", linestyle="--")
    axes[1].set_ylabel("Arm effort / limit")
    axes[1].legend(fontsize=7, ncol=2)
    axes[2].plot(times, [r["gripper_width_m"] * 1000 for r in rows], label="Measured")
    axes[2].plot(times, [r["gripper_reference_m"] * 1000 for r in rows], "--", label="Reference")
    axes[2].set_ylabel("Jaw opening (mm)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].legend()
    for axis in axes:
        axis.axvspan(2, 2.4, color="orange", alpha=0.2)
        axis.grid(alpha=0.2)
    fig.suptitle(f"{report['robot'].title()}: gravity hold, 1 N wrist push (orange), recovery, gripper")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=130)
    plt.close(fig)


def run(args):
    robot = load_robot(args.robot)
    compatibility = preflight(robot)
    model, data = robot.model, robot.data
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    require(model.neq == 1, "The test must use the task model with only its finger equality")
    tool_reference = robot.tool_position
    wrist = model.body("wrist_link" if args.robot == "forte" else "hand").id
    # Forte fuses static bodies into world; Panda retains its named table body.
    environment = {0, *(model.body(f"cube_{color}").id for color in ("red", "green", "blue"))}
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if table >= 0:
        environment.add(table)
    robot_geoms = np.array([body not in environment for body in model.geom_bodyid])
    geom_names = [model.geom(i).name for i in range(model.ngeom)]
    peak_effort = np.zeros(len(robot.home))
    peak_speed = np.zeros(len(robot.home))
    peak_error = np.zeros(len(robot.home))
    saturation = np.zeros(len(robot.home), dtype=int)
    peak_tool = peak_penetration = peak_gripper_force = peak_gripper_speed = peak_mimic = 0.0
    initial_error = recovery_error = final_error = 0.0
    settled_joint_error = 0.0
    observed_push = 0.0
    rows = []
    phase_previous = None
    dt = model.opt.timestep
    count = round(14 / dt)
    context = nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(model, data)
    report = {"status": "INCOMPLETE", "robot": args.robot, "joint_names": list(robot.joint_names),
              "mujoco_version": mujoco.__version__,
              "scope": "unbraced home hold, wrist disturbance recovery, empty gripper; no IK or payload",
              "scene_sha256": hashlib.sha256(SCENES[args.robot].read_bytes()).hexdigest(),
              "gravity_m_per_s2": model.opt.gravity.tolist(), "timestep_s": dt,
              "equality_count": model.neq, "mocap_count": model.nmocap,
              "arm_effort_limits_nm": robot.effort_limits.tolist(),
              "push_force_n": [0, 0, -1], "push_interval_s": [2, 2.4]}
    # Forte has explicit gravity feedforward. Panda compatibility preserves its
    # original unmodified servo gains and their measured 6.72 mm home sag.
    report["thresholds"] = {"settled_tool_error_m": 0.002 if args.robot == "forte" else 0.010,
                            "settled_joint_error_rad": float(np.deg2rad(0.25 if args.robot == "forte" else 1.0)),
                            "transient_tool_error_m": 0.020}
    if compatibility is not None:
        report["legacy_compatibility"] = compatibility
    if isinstance(robot, ForteRobot):
        report["arm_configuration"] = robot.config
        report["gripper_configuration"] = robot.gripper.config
    print(f"{args.robot.upper()}: real gravity, {len(robot.home)} arm joints, no arm-lock fixture", flush=True)
    try:
        with context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = (0.4, 0, 0.3)
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 140, -25
                    if args.robot == "forte":
                        viewer.opt.geomgroup[0] = int(args.collision_geoms)
                        viewer.opt.geomgroup[1] = int(not args.collision_geoms)
            wall_start = time.perf_counter()
            for index in range(count):
                t = index * dt
                if viewer is not None and not viewer.is_running():
                    raise InterruptedError("Viewer closed before completing the hold test")
                phase = ("HOLD HOME" if t < 2 else "1 N DOWNWARD WRIST PUSH" if t < 2.4 else
                         "RECOVER / HOLD" if t < 7 else "CLOSE EMPTY GRIPPER / HOLD ARM" if t < 10 else
                         "OPEN GRIPPER / HOLD ARM" if t < 12 else "FINAL HOLD")
                if phase != phase_previous:
                    print(f"{t:4.1f}s: {phase}", flush=True)
                    phase_previous = phase
                width, driver_speed = reference(t, robot.open_width, robot.closed_width)
                robot.set_gripper_width(width, driver_speed if isinstance(robot, ForteRobot) else 0.0)
                data.xfrc_applied[:] = 0
                if 2 <= t < 2.4:
                    data.xfrc_applied[wrist, 2] = -1
                robot.step()  # includes controller updates during all waits and gripper phases
                error = np.abs(robot.arm_position - robot.home)
                tool_error = float(np.linalg.norm(robot.tool_position - tool_reference))
                peak_error = np.maximum(peak_error, error)
                peak_effort = np.maximum(peak_effort, np.abs(robot.last_effort))
                peak_speed = np.maximum(peak_speed, np.abs(robot.arm_velocity))
                saturation += np.abs(robot.last_raw_effort) > robot.effort_limits + 1e-9
                peak_tool = max(peak_tool, tool_error)
                peak_gripper_force = max(peak_gripper_force, abs(robot.last_gripper_effort))
                require(np.all(np.abs(robot.last_effort) <= robot.effort_limits + 1e-6), "Arm torque cap exceeded")
                require(np.all(robot.arm_position >= robot.lower - 0.001)
                        and np.all(robot.arm_position <= robot.upper + 0.001), "Arm travel limit exceeded")
                require(np.all(np.abs(robot.arm_velocity) <= robot.speed_limits + 0.001), "Arm speed limit exceeded")
                if 1 <= t < 2:
                    initial_error = max(initial_error, tool_error)
                    settled_joint_error = max(settled_joint_error, float(np.max(error)))
                if 2 <= t < 3:
                    observed_push = max(observed_push, tool_error)
                if 6 <= t < 7:
                    recovery_error = max(recovery_error, tool_error)
                    settled_joint_error = max(settled_joint_error, float(np.max(error)))
                if t >= 13:
                    final_error = max(final_error, tool_error)
                    settled_joint_error = max(settled_joint_error, float(np.max(error)))
                for contact in data.contact:
                    a, b = int(contact.geom1), int(contact.geom2)
                    peak_penetration = max(peak_penetration, -float(contact.dist))
                    if (robot_geoms[a] or robot_geoms[b]) and contact.dist < -1e-5:
                        raise ValueError(f"Unexpected robot collision: {geom_names[a]} / {geom_names[b]}")
                if isinstance(robot, ForteRobot):
                    grip = robot.gripper
                    measured_width, center = grip.geometry(data)
                    speed = float(np.max(np.abs(data.qvel[[grip.driver_dof, grip.follower_dof]])))
                    peak_gripper_speed = max(peak_gripper_speed, speed)
                    mimic = abs(float(data.qpos[grip.driver_qpos] + data.qpos[grip.follower_qpos]))
                    peak_mimic = max(peak_mimic, mimic)
                    require(center < 0.0001 and mimic < 0.0001, "Gripper centering/coupling failed")
                    require(speed <= grip.config["speed_limit_m_per_s"] + 0.001, "Gripper speed limit exceeded")
                    require(abs(robot.last_gripper_effort) <= grip.config["effort_limit_n"], "Gripper force cap exceeded")
                    allowed = np.r_[robot.dof_indices, grip.driver_dof]
                    require(np.all(data.qfrc_applied[np.setdiff1d(np.arange(model.nv), allowed)] == 0),
                            "Compensation/control leaked into cube or follower coordinates")
                else:
                    measured_width = float(np.sum(data.qpos[robot.finger_positions]))
                    peak_gripper_speed = max(peak_gripper_speed, float(np.max(np.abs(data.qvel[robot.finger_dofs]))))
                    peak_mimic = max(peak_mimic, float(abs(np.diff(data.qpos[robot.finger_positions])[0])))
                if index % max(1, round(0.01 / dt)) == 0:
                    row = {"time_s": float(data.time), "phase": phase, "tool_error_m": tool_error,
                           "arm_error_max_rad": float(np.max(error)), "gripper_width_m": measured_width,
                           "gripper_reference_m": width, "gripper_effort_n": robot.last_gripper_effort}
                    row.update({f"{name}_effort_fraction": float(value) for name, value in
                                zip(robot.joint_names, np.abs(robot.last_effort) / robot.effort_limits)})
                    rows.append(row)
                if viewer is not None and index % max(1, round(1 / 60 / dt)) == 0:
                    viewer.sync()
                    time.sleep(max(0, wall_start + (index + 1) * dt - time.perf_counter()))
            data.xfrc_applied[:] = 0
            require(max(initial_error, recovery_error, final_error) < report["thresholds"]["settled_tool_error_m"],
                    "Settled tool error exceeds the configured test tolerance")
            require(settled_joint_error < report["thresholds"]["settled_joint_error_rad"],
                    "Settled arm error exceeds the configured test tolerance")
            require(peak_tool < 0.020, "Transient tool displacement exceeds 20 mm")
            require(not np.any(saturation), "Arm saturated during this home-pose protocol")
            if isinstance(robot, ForteRobot):
                require(observed_push > 0.0001, "No measurable push response; check for hidden constraints")
                require(abs(measured_width - robot.open_width) < 0.0002, "Gripper did not reopen")
            report.update({"status": "PASS", "simulated_seconds": float(data.time),
                           "peak_tool_displacement_m": peak_tool,
                           "initial_settled_tool_error_m": initial_error,
                           "recovered_tool_error_m": recovery_error,
                           "final_settled_tool_error_m": final_error,
                           "settled_joint_error_max_rad": settled_joint_error,
                           "peak_arm_error_rad": peak_error.tolist(), "peak_arm_effort_nm": peak_effort.tolist(),
                           "peak_arm_speed_rad_per_s": peak_speed.tolist(), "saturated_steps": saturation.tolist(),
                           "peak_gripper_effort_n": peak_gripper_force,
                           "peak_gripper_speed_m_per_s": peak_gripper_speed,
                           "peak_gripper_coupling_error_m": peak_mimic,
                           "peak_contact_penetration_m": peak_penetration,
                           "mujoco_warning_count": sum(int(w.number) for w in data.warning)})
    except (ValueError, RuntimeError, InterruptedError) as exc:
        report.update({"status": "FAIL", "reason": str(exc), "simulated_seconds": float(data.time)})
        save_outputs(args.report, report, rows)
        raise
    save_outputs(args.report, report, rows)
    print(f"PASS: 14 s gravity hold/recovery/gripper protocol; peak displacement {peak_tool * 1000:.3f} mm")
    print(f"Settled error after push {recovery_error * 1000:.6f} mm; peak torque {max(peak_effort):.3f} Nm")
    print(f"Report / plot / trace: {args.report.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("forte", "panda"), default="forte")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--viewer", action="store_true")
    mode.add_argument("--headless", action="store_true", help="Default; no window or wall-clock pacing")
    parser.add_argument("--collision-geoms", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.collision_geoms and not args.viewer:
        parser.error("--collision-geoms requires --viewer")
    if args.report is None:
        args.report = PROJECT / "outputs" / "diagnostics" / args.robot / "gravity_hold.json"
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError, InterruptedError) as exc:
        parser.exit(1, f"Robot check failed/incomplete: {exc}\n")


if __name__ == "__main__":
    main()
