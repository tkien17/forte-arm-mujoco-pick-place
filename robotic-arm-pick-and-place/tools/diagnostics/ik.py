"""Step 4: screened full-pose IK and open-gripper reaching; no picking yet."""

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
from controllers.ik import downward_quaternion, rotation_error
from controllers.reaching import ReachPlanner
from robots.factory import load_robot, SCENES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(report, rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if rows:
        with gzip.open(path.with_suffix(".csv.gz"), "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for point in report["workspace_grid"]:
        axes[0].scatter(point["x_m"], point["y_m"], c="green" if point["passed"] else "red", marker="+")
    for task in report["tasks"]:
        cube, target = task["cube_m"], task["target_m"]
        axes[0].scatter(cube[0], cube[1], c=task["color"], marker="s", s=65)
        axes[0].scatter(target[0], target[1], c=task["color"], marker="x", s=65)
        axes[0].plot([cube[0], target[0]], [cube[1], target[1]], color=task["color"], alpha=0.4)
    axes[0].set(xlabel="World X (m)", ylabel="World Y (m)", title="Screened samples (+), cubes (square), pads (x)")
    axes[0].set_aspect("equal")
    if rows:
        times = [r["time_s"] for r in rows]
        axes[1].plot(times, [r["tracking_error_m"] * 1000 for r in rows], label="Tool/reference error")
        axes[1].axhline(2, color="gray", linestyle="--", label="Settled tolerance")
        axes[1].set(xlabel="Simulation time (s)", ylabel="Position error (mm)", title="Open-hand reaching under gravity")
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].text(0.05, 0.7, "Static validation only\nRun without --validate-only for physics checks.")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=130)
    plt.close(fig)


def snapshot(robot, destination):
    import imageio.v2 as imageio
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(robot.model, height=480, width=640) as renderer:
        renderer.update_scene(robot.data, camera="demo_cam", scene_option=option)
        imageio.imwrite(destination, renderer.render())


def run(args):
    robot = load_robot("forte")
    planner = ReachPlanner(robot)
    # Any solver/planner operation must leave the complete live integration state alone.
    before = {name: getattr(robot.data, name).copy() for name in
              ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")}
    before_time = float(robot.data.time)
    paths_by_color, tasks = {}, []
    for color in ("green", "red", "blue"):
        paths, metadata = planner.plan_task(color)
        paths_by_color[color] = paths
        tasks.append(metadata)
        print(f"SCREENED {color}: downward yaw {metadata['yaw_deg']:g} deg; "
              f"{len(paths)} motions, {metadata['duration_s']:.1f} simulated seconds", flush=True)
    grid = planner.grid()
    require(all(p["passed"] for p in grid), "Some configured workspace grid samples failed screening")
    impossible = planner.ik.solve([3, 0, 1], downward_quaternion(), robot.home)
    require(not impossible.success, "Unreachable target was incorrectly accepted")
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(robot.data, name), value)
    require(robot.data.time == before_time, "IK/planning advanced live simulation time")
    endpoint_pe, endpoint_re = 0.0, 0.0
    for paths in paths_by_color.values():
        for path in paths:
            position, quaternion = planner.ik.pose(path.nodes[-1])
            endpoint_pe = max(endpoint_pe, float(np.linalg.norm(position - path.desired_position)))
            endpoint_re = max(endpoint_re, float(np.linalg.norm(rotation_error(path.desired_quaternion, quaternion))))
    require(endpoint_pe <= 0.0005 and endpoint_re <= np.deg2rad(0.5), "Endpoint FK verification failed")
    report = {
        "status": "STATIC_PASS", "scope": "sampled empty-gripper pose/path checks; no grasp or payload",
        "mujoco_version": mujoco.__version__, "workspace_configuration": planner.config,
        "arm_configuration": robot.config,
        "scene_sha256": hashlib.sha256(SCENES["forte"].read_bytes()).hexdigest(),
        "robot_sha256": hashlib.sha256(SCENES["forte"].with_name("forte.xml").read_bytes()).hexdigest(),
        "workspace_grid": grid, "tasks": tasks,
        "ik_endpoint_position_error_max_m": endpoint_pe,
        "ik_endpoint_rotation_error_max_rad": endpoint_re,
        "sampled_reference_states": planner.checked_samples,
        "minimum_reference_joint_margin_rad": planner.minimum_joint_margin,
        "minimum_reference_environment_clearance_m": planner.minimum_environment_clearance,
        "peak_reference_static_torque_fraction": planner.maximum_static_fraction,
        "unreachable_target_rejected": True, "live_state_unchanged_by_planning": True,
        "limitations": ["Discrete samples do not certify the entire continuous workspace.",
                        "Self-collision penetration is checked; the distance buffer applies to environment geoms.",
                        "Open-hand motion only; no object is lifted or moved by this demo."]}
    rows = []
    print(f"PASS: {len(grid)}/{len(grid)} grid samples; {planner.checked_samples} reference checks; "
          f"minimum environment clearance {planner.minimum_environment_clearance * 1000:.2f} mm", flush=True)
    if args.validate_only:
        save(report, rows, args.report)
        print(f"Static report: {args.report.resolve()}")
        return

    colors = ("green", "red", "blue") if args.task == "all" else (args.task,)
    context = nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(robot.model, robot.data)
    peak_tracking = peak_rotation = peak_gravity = peak_speed = peak_torque = peak_penetration = 0.0
    endpoint_errors = []
    saturation = np.zeros(6, dtype=int)
    cube_start = {c: robot.data.body(f"cube_{c}").xpos.copy() for c in ("red", "green", "blue")}
    dt = robot.model.opt.timestep
    frame_index = 0
    captured = False
    try:
        with context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = (0.4, 0, 0.3)
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 140, -25
                    viewer.opt.geomgroup[0] = int(args.collision_geoms)
                    viewer.opt.geomgroup[1] = int(not args.collision_geoms)
            wall_start = time.perf_counter()
            for color in colors:
                for path in paths_by_color[color]:
                    print(path.label, flush=True)
                    # The target remains active during every settling step.
                    steps = int(np.ceil((path.duration + 0.75) / dt))
                    settled_position = settled_rotation = 0.0
                    for index in range(steps):
                        if viewer is not None and not viewer.is_running():
                            raise InterruptedError("Viewer closed before completing the reaching demo")
                        t = (index + 1) * dt
                        qref, dqref = path.sample(t)
                        robot.set_arm_target(qref, dqref)
                        robot.set_gripper_width(robot.open_width)
                        robot.step()
                        reference_position, reference_quaternion = planner.ik.pose(qref)
                        actual_quaternion = np.zeros(4)
                        mujoco.mju_mat2Quat(actual_quaternion, robot.data.site_xmat[robot.site_id])
                        pe = float(np.linalg.norm(robot.tool_position - reference_position))
                        re = float(np.linalg.norm(rotation_error(reference_quaternion, actual_quaternion)))
                        peak_tracking, peak_rotation = max(peak_tracking, pe), max(peak_rotation, re)
                        peak_torque = max(peak_torque, float(np.max(np.abs(robot.last_effort))))
                        peak_speed = max(peak_speed, float(np.max(np.abs(robot.arm_velocity))))
                        peak_gravity = max(peak_gravity, float(np.max(np.abs(robot.last_gravity) / robot.effort_limits)))
                        saturation += np.abs(robot.last_raw_effort) > robot.effort_limits + 1e-9
                        require(pe < 0.010 and re < np.deg2rad(5), "Excessive dynamic pose tracking error")
                        require(np.all(np.abs(robot.arm_velocity) <= robot.speed_limits), "Arm speed bound exceeded")
                        require(np.all(np.abs(robot.last_effort) <= robot.effort_limits), "Torque bound exceeded")
                        require(np.all(robot.arm_position >= robot.lower)
                                and np.all(robot.arm_position <= robot.upper), "Measured joint limit exceeded")
                        for contact in robot.data.contact:
                            a, b = int(contact.geom1), int(contact.geom2)
                            peak_penetration = max(peak_penetration, -float(contact.dist))
                            if (planner.ik.robot_geoms[a] or planner.ik.robot_geoms[b]) and contact.dist < -1e-5:
                                raise ValueError(f"Unexpected robot contact: {robot.model.geom(a).name}/{robot.model.geom(b).name}")
                        if t > path.duration + 0.50:
                            settled_position = max(settled_position, float(np.linalg.norm(robot.tool_position - path.desired_position)))
                            settled_rotation = max(settled_rotation, float(np.linalg.norm(rotation_error(path.desired_quaternion, actual_quaternion))))
                        if frame_index % 10 == 0:
                            accepted, reason, _, _ = planner.ik.screen(robot.arm_position)
                            require(accepted, f"Measured pose screening failed: {reason}")
                            rows.append({"time_s": float(robot.data.time), "path": path.label,
                                         "tracking_error_m": pe, "rotation_error_rad": re,
                                         "max_effort_nm": float(np.max(np.abs(robot.last_effort))),
                                         "reference_x_m": float(reference_position[0]),
                                         "reference_y_m": float(reference_position[1]),
                                         "reference_z_m": float(reference_position[2]),
                                         "actual_x_m": float(robot.tool_position[0]),
                                         "actual_y_m": float(robot.tool_position[1]),
                                         "actual_z_m": float(robot.tool_position[2])})
                        if viewer is not None and frame_index % 17 == 0:
                            # White point is the current tool reference; it has no physics.
                            with viewer.lock():
                                viewer.user_scn.ngeom = 1
                                mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                                                   np.array([0.004] * 3), reference_position,
                                                   np.eye(3).reshape(-1), np.array([1, 1, 1, 0.7], dtype=np.float32))
                            viewer.sync()
                            time.sleep(max(0, wall_start + robot.data.time - time.perf_counter()))
                        frame_index += 1
                    require(settled_position < 0.002 and settled_rotation < np.deg2rad(2),
                            f"{path.label}: failed settled pose tolerance")
                    endpoint_errors.append({"path": path.label, "position_error_m": settled_position,
                                            "rotation_error_rad": settled_rotation})
                    if args.snapshot and not captured and "descend" in path.label:
                        snapshot(robot, args.snapshot)
                        captured = True
            cube_drift = max(float(np.linalg.norm(robot.data.body(f"cube_{c}").xpos - point))
                             for c, point in cube_start.items())
            require(cube_drift < 0.001, "An open-hand reach disturbed a cube")
            require(not np.any(saturation), "Torque saturation during reaching")
            report.update({"status": "PASS", "executed_tasks": list(colors),
                           "simulated_seconds": float(robot.data.time),
                           "peak_tracking_error_m": peak_tracking, "peak_rotation_tracking_error_rad": peak_rotation,
                           "peak_arm_effort_nm": peak_torque, "peak_arm_speed_rad_per_s": peak_speed,
                           "peak_static_gravity_fraction": peak_gravity, "saturated_steps": saturation.tolist(),
                           "endpoint_errors": endpoint_errors, "maximum_cube_drift_m": cube_drift,
                           "peak_contact_penetration_m": peak_penetration,
                           "mujoco_warning_count": sum(int(w.number) for w in robot.data.warning)})
    except (ValueError, RuntimeError, InterruptedError) as exc:
        report.update({"status": "FAIL", "reason": str(exc), "simulated_seconds": float(robot.data.time)})
        save(report, rows, args.report)
        raise
    save(report, rows, args.report)
    print(f"PASS: {len(colors)} open-hand route(s), {robot.data.time:.1f} simulated seconds; "
          f"peak tracking error {peak_tracking * 1000:.3f} mm; torque {peak_torque:.3f} Nm")
    print(f"Report / plot / trace: {args.report.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--viewer", action="store_true")
    mode.add_argument("--headless", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--task", choices=("green", "red", "blue", "all"), default="green")
    parser.add_argument("--collision-geoms", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.collision_geoms and not args.viewer:
        parser.error("--collision-geoms requires --viewer")
    if args.validate_only and args.viewer:
        parser.error("--validate-only does not run a viewer")
    if args.report is None:
        name = "workspace_static" if args.validate_only else f"ik_reaching_{args.task}"
        args.report = PROJECT / "outputs" / "diagnostics" / "forte" / f"{name}.json"
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, AssertionError, InterruptedError) as exc:
        parser.exit(1, f"IK validation failed/incomplete: {exc}\n")


if __name__ == "__main__":
    main()
