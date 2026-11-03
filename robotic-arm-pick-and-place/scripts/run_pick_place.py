"""Run the existing Panda task or the Forte physical pick-and-place task."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import mujoco

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from controllers.scripted_controller import TASKS, make_controller
from robots.factory import load_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("panda", "forte"), default="panda")
    parser.add_argument("--task", choices=("green", "red", "blue", "all"), default="all")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer and exit after the task")
    parser.add_argument("--randomize", action="store_true", help="Jitter cube spawn positions at reset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--collision-geoms", action="store_true", help="Show collision primitives in the viewer")
    args = parser.parse_args()
    if args.headless and args.collision_geoms:
        parser.error("--collision-geoms requires a viewer")

    robot = load_robot(args.robot)
    ctrl = make_controller(robot, seed=args.seed)
    ctrl.reset(randomize=args.randomize)
    tasks = TASKS if args.task == "all" else [(f"cube_{args.task}", f"target_{args.task}")]
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as viewer_module
        context = viewer_module.launch_passive(robot.model, robot.data)
    results = []
    with context as viewer:
        if viewer is not None:
            with viewer.lock():
                viewer.cam.lookat[:] = (0.4, 0, 0.3)
                viewer.cam.distance = 1.4
                viewer.cam.azimuth = 140
                viewer.cam.elevation = -25
                if args.robot == "forte":
                    viewer.opt.geomgroup[0] = int(args.collision_geoms)
                    viewer.opt.geomgroup[1] = int(not args.collision_geoms)
                else:
                    # Panda uses groups 2/3 for CAD/collisions; environment is 0.
                    viewer.opt.geomgroup[2] = int(not args.collision_geoms)
                    viewer.opt.geomgroup[3] = int(args.collision_geoms)
        try:
            ctrl._settle(300, viewer)
            for cube, target in tasks:
                ok = ctrl.pick_and_place(cube, target, viewer)
                result = getattr(ctrl, "last_result", None)
                results.append(result or {"cube": cube, "target": target, "success": bool(ok)})
                print(f"{cube} -> {target}: {'SUCCESS' if ok else 'FAILED'}", flush=True)
                if not ok:
                    break
        except (ValueError, RuntimeError, InterruptedError) as exc:
            results.append({"success": False, "reason": str(exc)})

        passed = len(results) == len(tasks) and all(r["success"] for r in results)
        report = args.report or PROJECT / "outputs" / "runs" / args.robot / f"pick_place_{args.task}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"robot": args.robot, "mujoco_version": mujoco.__version__,
                                      "status": "PASS" if passed else "FAIL", "seed": args.seed,
                                      "randomized": args.randomize, "simulated_seconds": float(robot.data.time),
                                      "results": results}, indent=2) + "\n", encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'}: {sum(r['success'] for r in results)}/{len(tasks)} placed; report: {report}", flush=True)
        if viewer is not None and passed:
            print("Close the viewer to exit.", flush=True)
            while viewer.is_running():
                try:
                    ctrl._settle(16, viewer)
                except InterruptedError:
                    break
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
