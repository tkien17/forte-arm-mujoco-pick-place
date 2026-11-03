# Customized Forte integration

Steps 1-5 are implemented. Pause for user inspection before step 6. This guide
consolidates the previous plan and stage notes; the project README contains
the current folder map and runnable commands.

## Robot and model

The supplied `ForteME.urdf` defines six movable arm hinges:
`shoulder_yaw`, `shoulder_pitch`, `shoulder_roll`, `elbow_pitch`,
`lower_arm_roll`, and `wrist_pitch`. `wrist_to_gripper` is fixed. Two finger
sliders are constrained by `q_right = -q_left`, giving one independent
opening input. Thus the supplied model has six arm DoF and one gripper DoF.

`assets/forte_arm/source/` contains the original URDF and ten STL files,
copied byte-for-byte. `source_manifest.json` records SHA-256 checksums and
provenance. `tools/build_forte_scene.py` imports the URDF in memory, retains
visuals and collision primitives, adds one mimic constraint and a tool site,
and writes portable `forte.xml` and `scene.xml`. It leaves source files intact.
The scene reuses the Panda environment with a Forte-specific table/layout.
The robot base is fixed. There are no arm-lock constraints in the task scene,
no grasp weld, and no robot actuators; forces are applied by the control backend.

The 1 ms implicitfast simulation uses gravity `[0, 0, -9.81]`. Step 5 uses
elliptic friction cones with `impratio=10` to reduce slow tangential contact
slip. This follows [MuJoCo's contact guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip).
The source collision geometry, inertias, and friction coefficients remain
unchanged. The gripper-only fixture is a separate diagnostic model.

## Configuration and target poses

`configs/forte.json` is the single configuration file:

- `arm`: home angles and per-joint PD gains. Effort/speed limits are read from
  the original URDF: 10 Nm per arm joint; slider force capped at 10 N.
- `gripper`: wrist-local tool offset `[0.227, 0.0285, 0]` m, open width
  108 mm, empty-close width 10 mm, grasp reference 30 mm. For the 40 mm cube,
  contact blocks further closure at approximately 39.7 mm. The reference
  overclosure creates force; it does not shrink the cube or overwrite fingers.
- `workspace`: table center `[0.52, 0, 0.10]` m, half-size `[0.32, 0.40, 0.10]` m;
  cubes around X=0.42-0.45 m, Y=-0.10/0/+0.10 m; pads X=0.50-0.52 m;
  hover Z=0.27 m and release clearance 10 mm above the resting cube center.

The opening relation is `width = 0.069 - 2*q_left` in metres. The wrist-local
+X points along the fingers and +Y is the closing axis. A top-down orientation
maps local +X to world -Z. The planner tries configured yaw candidates and
arm seeds, then rejects invalid routes before execution.

The scene provides exact simulated cube positions. There is no camera-based
object detection or general grasp inference. The controller constructs tool
poses from the cube center, the calibrated grasp frame, a chosen downward
orientation, table height, and approach/release offsets. IK then solves six
arm angles. Position alone does not define a grasp; orientation and finger
opening are required. Following grasp, the measured cube-to-tool transform
is included in scratch collision planning and refreshed before lowering.

## Control and physical grasping

`RobotInterface` maps named joints instead of assuming array prefixes. Its
step method recomputes control before every physics step, including dwells.
Forte uses bounded PD with pure gravity feedforward computed at zero velocity
on a separate state. During transport, arm feedforward also compensates the
50 g cube's weight through a Jacobian; no compensation is applied to the free
cube itself. A single bounded force drives the left slider, with passive
right-slider mimic coupling.

`PoseIK` solves position and quaternion orientation on scratch state. It never
assigns live robot or cube positions. Path screening checks joint margins,
static torque reserve, self/environment collision, and environment clearance.
Quintic time scaling limits reference speed. A held cube is moved with the
predicted tool only in the scratch planning state, so its collisions are
screened too. Only distal finger-to-held-cube contacts are permitted.

`ForteController.pick_and_place` executes approach, descend, close, contact
verification, lift, carry, lower, release, retreat, and return home. It checks
bilateral contact, at least 35 mm lift, tool endpoint tracking, at most 3 mm
relative grasp slip, and a released stable cube within 20 mm XY of its pad.
It reports torque, driver force, contact penetration, slip, and numerical
warnings. Failures stop the entry script and return a nonzero exit code.
The original Panda `ScriptedController` and its main methods remain available.

## Completed stages and checks

| Stage | Implementation | Verification |
| --- | --- | --- |
| 1 | Portable MJCF and task scene | URDF/STL hashes; joint types/limits; source inertias/geometry; finite collision-free home; relative asset paths |
| 2 | Calibrated tool and one-input gripper | Supported fixture: open/close, mimic, bilateral contact, force/speed limits, physical release |
| 3 | Common interface and Forte gravity control | Unbraced 14 s hold, 1 N wrist disturbance/recovery, empty-gripper motion |
| 4 | Full-pose IK and screened workspace/routes | 12 configured grid samples; all three empty-hand physical rehearsals; unreachable/obstructed/overloaded requests rejected |
| 5 | Physical grasp and main task entry point | All three cubes lifted, transported, released, and stable at their pads; Panda green task regression; six IK/payload unit tests |

Stage 3 peak wrist deflection was 7.419 mm under the disturbance, recovering
to below 0.001 mm. Peak arm effort was 9.449 Nm without saturation.
Stage 4's all-color rehearsal took 140.7 simulated seconds with 0.133 mm peak
reference tracking error, 8.640 Nm peak arm effort, and no unexpected contact.
These are measured protocols and sampled routes, not a continuous-workspace
certificate. Generated JSON, plots, and traces are kept under `outputs/` and
ignored by Git. Current task results are `outputs/runs/forte/pick_place_all.json`.

The step 5 nominal three-cube run took 157.46 simulated seconds and placed
3/3 cubes successfully. Final XY errors were 0.287/0.205/0.283 mm for
green/red/blue. Each cube lifted approximately 49.87 mm. Maximum measured
grasp slip was 0.715 mm, arm effort 8.640 Nm, gripper driver effort 2.910 N,
and robot contact penetration 0.166 mm. There were no saturated steps or
MuJoCo warnings. These are nominal-layout results, not randomized success rates.

The six unit tests cover full-pose convergence with unchanged live state,
quaternion sign equivalence, invalid/unreachable targets, obstruction,
static gravity overload, reordered robot/object DOFs, and held-object
planning that applies no force to free objects or extra attachment constraint.

## Step 6: remaining work after confirmation

1. Adapt the existing `move_arm.py` to use the common robot interface and a
   screened Forte rehearsal while retaining the Panda demo.
2. Adapt the existing `record_demo.py` and recording helper to use the selected
   robot/controller, simulation-time capture, renderer cleanup, and separate
   robot output paths. Record and inspect a real Forte task GIF/video.
3. Adapt the existing `eval/evaluate.py` to evaluate both robots, retain seeded
   resets, and save per-robot measurements. Forte randomization starts with
   bounded +/-8 mm XY spawn perturbations inside the configured sampled range;
   it must not be represented as broad workspace reliability.
4. Run seeded multi-episode trials and Panda regression checks, report failures
   and measured rates, and verify all documented commands from either folder.

Acceptance: all four original entry scripts work for Forte, a recorded sequence
shows physical pick/release, and reproducible randomized measurements are
reported with the tested spawn range and sample count. Expand workspace or
success claims only after additional validation. No real hardware control is
included in this integration.
