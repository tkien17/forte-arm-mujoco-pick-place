# MuJoCo robotic arm pick-and-place

A scripted controller picks up three colored cubes and places each on its
matching pad. The original Panda implementation and the customized Forte
implementation share the task entry point. Both use simulated contact grasps.

Panda has seven arm joints. The supplied Forte URDF has **six arm joints plus
one coupled gripper input**; its wrist-to-gripper joint is fixed.

## Structure

```text
robotic-arm-pick-and-place/
  assets/          Robot models, task scenes, original Forte URDF/STLs
  configs/        forte.json: arm gains, gripper settings, workspace
  robots/         Robot interface, named joints, control backends, scene loading
  controllers/    Scripted tasks, full-pose Forte IK, screened reaching paths
  scripts/        User entry points: move_arm, run_pick_place, record_demo
  eval/           Randomized task evaluation
  utils/          Recording helper
  tools/          check.py entry point, model builder, diagnostics/
  tests/          IK and payload planning regression tests
  docs/           forte.md: integration details and stage checklist
  outputs/        Generated diagnostics and task reports (ignored by Git)
  media/          Existing demo GIF and success-rate image
```

`controllers/scripted_controller.py` retains the original Panda
`ScriptedController` and adds `ForteController` and `make_controller`.
`controllers/ik.py` solves Forte tool position **and orientation** using only
the six named arm joints. `controllers/reaching.py` checks joint limits,
static torque, collisions, and clearance along each proposed path.
`robots/forte.py` applies bounded PD and gravity/payload feedforward each
simulation step. `robots/forte_gripper.py` drives one slider; the other follows
through the mimic constraint. The main controller functions remain in code;
test fixtures and generated reports live separately under `tools/` and `outputs/`.

## Setup

Tested locally with Python 3.14 and MuJoCo 3.17. From this repository's outer
folder, the existing Windows environment can be used directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r robotic-arm-pick-and-place/requirements.txt
```

For a fresh environment, create a virtual environment first. The robot assets
are included; the original external Downloads folder is not required.
Windows/Linux use Python for the live viewer. macOS requires `mjpython` for
interactive viewing; offscreen and headless checks use Python.

## Forte: run and inspect step 5

Run these commands from the **outer repository folder**:

```powershell
# Watch one physical grasp, lift, transfer, release, and return home.
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/scripts/run_pick_place.py --robot forte --task green

# Run all three without a window; exits with a failure code if any task fails.
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/scripts/run_pick_place.py --robot forte --task all --headless
```

The live viewer runs at approximately simulation speed. A full three-cube
run takes roughly three simulated minutes. Watch the cube stay between the
fingers during lift/transfer, leave the fingers during release, settle on its
matching pad, and remain there as the arm returns home. Close the viewer to exit.
Use `--collision-geoms` to inspect collision primitives, `--task red`/`blue`
for a single color, or `--randomize --seed 0` for small spawn perturbations.
Reports go to `outputs/runs/forte/pick_place_<task>.json`.

Panda remains the default. For its original task:

```powershell
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/scripts/run_pick_place.py --robot panda --task green
```

## One diagnostic entry point

```powershell
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/tools/check.py model --check
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/tools/check.py model --preview
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/tools/check.py gripper --viewer
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/tools/check.py hold --robot forte --viewer
.\.venv\Scripts\python.exe robotic-arm-pick-and-place/tools/check.py ik --viewer --task green
.\.venv\Scripts\python.exe -m unittest discover -s robotic-arm-pick-and-place/tests -v
```

`model --preview` pauses physics. Gripper calibration uses temporary arm locks
and a supported cube. `hold` uses the real unbraced task scene. `ik` rehearses
the route with the gripper open. Only `run_pick_place.py` executes physical
Forte grasping. Run `tools/check.py model` without `--check` to rebuild the
MJCF from the included source assets and configuration.


See [the Forte guide](docs/forte.md) for the completed stages, configuration,
validation limits, and the final stage's acceptance criteria.

## License

Project code: MIT. The original Panda model comes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0; see `assets/franka_emika_panda/LICENSE`). Forte source provenance
and checksums are in `assets/forte_arm/source_manifest.json`; the Panda asset
license does not establish redistribution rights for the supplied Forte CAD.
