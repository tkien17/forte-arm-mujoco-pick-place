"""Build, check, and preview the Forte scene before implementing motion control.

All paths are resolved relative to this repository. The preview holds a fixed
configuration with physics paused; it is a model inspection, not a controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / "assets" / "forte_arm"
SOURCE = ASSETS / "source"
URDF = SOURCE / "ForteME.urdf"
MANIFEST = ASSETS / "source_manifest.json"
ROBOT = ASSETS / "forte.xml"
SCENE = ASSETS / "scene.xml"
PANDA_SCENE = PROJECT / "assets" / "franka_emika_panda" / "scene.xml"
ARM_JOINTS = (
    "shoulder_yaw", "shoulder_pitch", "shoulder_roll", "elbow_pitch",
    "lower_arm_roll", "wrist_pitch",
)
LEFT = "l_gripper_linear_joint"
RIGHT = "r_gripper_linear_joint"
CONFIG = json.loads((PROJECT / "configs" / "forte.json").read_text(encoding="utf-8"))
ARM_CONFIG = CONFIG["arm"]
HOME = tuple(ARM_CONFIG["joints"][name]["home_rad"] for name in ARM_JOINTS)
GRIPPER_CONFIG = CONFIG["gripper"]
TOOL_OFFSET = tuple(GRIPPER_CONFIG["tool_offset_m"])
WORKSPACE = CONFIG["workspace"]
COLORS = ("red", "green", "blue")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mesh_filenames(root: ET.Element) -> list[str]:
    names = sorted({mesh.attrib["filename"] for mesh in root.findall(".//mesh")})
    if any(Path(name).name != name or Path(name).suffix.lower() != ".stl" for name in names):
        raise ValueError("Expected flat STL filenames in the supplied Forte URDF")
    return names


def import_source(folder: Path) -> None:
    """Vendor only the supplied URDF and its referenced meshes, byte for byte."""
    folder = folder.resolve(strict=True)
    source_urdf = folder / "ForteME.urdf"
    names = [source_urdf.name, *mesh_filenames(ET.parse(source_urdf).getroot())]
    records = []
    for name in names:
        original = folder / name
        checksum = digest(original)
        destination = SOURCE / name
        if destination.exists() and digest(destination) != checksum:
            raise ValueError(f"Refusing to overwrite a changed vendored source: {destination}")
        records.append({"file": name, "bytes": original.stat().st_size, "sha256": checksum})
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (SOURCE / name).exists():
            shutil.copy2(folder / name, SOURCE / name)
    manifest = {
        "description": "User-supplied Forte URDF and ten CAD meshes; copied without modification",
        "source_directory": str(folder),
        "files": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def check_source() -> ET.Element:
    if not URDF.exists() or not MANIFEST.exists():
        raise ValueError("Import the source first with --import-from PATH_TO_FORTE_FOLDER")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root = ET.parse(URDF).getroot()
    expected = {URDF.name, *mesh_filenames(root)}
    records = manifest["files"]
    if len(records) != len(expected) or {entry["file"] for entry in records} != expected:
        raise ValueError("Source manifest does not match the URDF mesh references")
    for entry in records:
        path = SOURCE / entry["file"]
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Source integrity check failed: {path}")
    if len(expected) != 11:
        raise ValueError("Expected one URDF and ten meshes")
    return root


def write_xml(root: ET.Element, destination: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def build() -> None:
    root = check_source()
    compiler = root.find("mujoco/compiler")
    if compiler is None:
        raise ValueError("The supplied Forte URDF must retain its MuJoCo compiler extension")
    compiler.set("discardvisual", "false")
    compiler.set("meshdir", SOURCE.as_posix())

    # Normalize the mimic in memory so importers cannot create a duplicate.
    # Never edit the source URDF or change either finger's CAD transform.
    mimic = root.find(f"joint[@name='{RIGHT}']/mimic")
    if mimic is None or mimic.attrib != {"joint": LEFT, "multiplier": "-1", "offset": "0"}:
        raise ValueError("Expected the supplied right = -left gripper mimic relation")
    root.find(f"joint[@name='{RIGHT}']").remove(mimic)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.modelname = "forte_arm"
    # Save authored precision; compiled canonical output rounds to six digits.
    spec.compiler.savecompiled = False
    equality = spec.add_equality()
    equality.name = "gripper_mimic"
    equality.type = mujoco.mjtEq.mjEQ_JOINT
    equality.name1, equality.name2 = RIGHT, LEFT
    equality.data[:5] = (0, -1, 0, 0, 0)
    equality.solref[:] = (0.002, 1)
    equality.solimp[:] = (0.999, 0.999, 0.001, 0.5, 2)
    spec.option.timestep = 0.001
    spec.option.gravity[:] = (0, 0, -9.81)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Harder tangential constraints reduce slow slip in contact grasps.
    # Keep the supplied geometry, masses and friction coefficients unchanged.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.body(GRIPPER_CONFIG["tool_body"]).add_site(
        name=GRIPPER_CONFIG["tool_site"], pos=TOOL_OFFSET, size=(0.005, 0, 0), rgba=(0, 1, 0, 1),
    )
    spec.compile()
    robot_root = ET.fromstring(spec.to_xml())
    robot_root.find("compiler").set("meshdir", "source")
    for mesh in robot_root.findall("asset/mesh"):
        mesh.set("file", Path(mesh.attrib["file"]).name)
    write_xml(robot_root, ROBOT)

    # Reuse the Panda environment, leaving its original files intact.
    scene_root = ET.parse(PANDA_SCENE).getroot()
    scene_root.set("model", "forte_pick_and_place_scene")
    scene_root.find("include").set("file", "forte.xml")
    table = scene_root.find("worldbody/body[@name='table']")
    table.set("pos", " ".join(map(str, WORKSPACE["table_center_m"])))
    table.find("geom").set("size", " ".join(map(str, WORKSPACE["table_half_size_m"])))
    table_z = WORKSPACE["table_center_m"][2] + WORKSPACE["table_half_size_m"][2]
    for color in COLORS:
        scene_root.find(f"worldbody/body[@name='cube_{color}']").set(
            "pos", " ".join(map(str, WORKSPACE["cube_spawn_m"][color])))
        scene_root.find(f"worldbody/site[@name='target_{color}']").set(
            "pos", " ".join(map(str, [*WORKSPACE["target_xy_m"][color], table_z + 0.001])))
    # Keep the environment visible when hiding the robot collision primitives
    # (group 0) or its CAD meshes (group 1) in the inspection viewer.
    for geom in scene_root.findall("worldbody//geom"):
        geom.set("group", "2")
    write_xml(scene_root, SCENE)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    for name, value in zip(ARM_JOINTS, HOME):
        data.qpos[model.joint(name).qposadr[0]] = value
    # Preview the cubes resting on the table; this does not change qpos0.
    for color in COLORS:
        joint_id = int(model.body(f"cube_{color}").jntadr[0])
        data.qpos[model.jnt_qposadr[joint_id] + 2] = table_z + model.geom(f"cube_{color}").size[2]
    keyframes = ET.SubElement(scene_root, "keyframe")
    ET.SubElement(keyframes, "key", {
        "name": "home", "qpos": " ".join(f"{value:.17g}" for value in data.qpos),
    })
    write_xml(scene_root, SCENE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check() -> tuple[mujoco.MjModel, mujoco.MjData]:
    root = check_source()
    original = mujoco.MjModel.from_xml_path(str(URDF))
    robot = mujoco.MjModel.from_xml_path(str(ROBOT))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    require((robot.nq, robot.nv, robot.njnt, robot.nu, robot.neq) == (8, 8, 8, 0, 1),
            "Robot must have 8 scalar coordinates, no actuators, and exactly one coupling")
    require((model.nq, model.nv, model.njnt, model.nu, model.neq) == (29, 26, 11, 0, 1),
            "Unexpected full-scene coordinates or constraints")
    for joint in root.findall("joint"):
        if joint.attrib["type"] == "fixed":
            continue
        name = joint.attrib["name"]
        for compiled in (robot, model):
            actual = compiled.joint(name)
            expected_type = (mujoco.mjtJoint.mjJNT_HINGE if name in ARM_JOINTS
                             else mujoco.mjtJoint.mjJNT_SLIDE)
            require(int(actual.type[0]) == int(expected_type), f"Wrong joint type: {name}")
            limits = joint.find("limit")
            np.testing.assert_allclose(actual.range, [float(limits.get("lower")),
                                                      float(limits.get("upper"))], atol=1e-10)
    for attribute in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
                      "jnt_axis", "jnt_pos", "jnt_range"):
        np.testing.assert_allclose(getattr(robot, attribute), getattr(original, attribute),
                                   rtol=1e-6, atol=1e-9, err_msg=f"Changed source {attribute}")
    require(robot.ngeom == original.ngeom and robot.nmesh == original.nmesh,
            "Robot geometry or mesh count changed")
    np.testing.assert_array_equal(robot.geom_type, original.geom_type)
    np.testing.assert_allclose(robot.geom_size, original.geom_size, atol=1e-9)
    np.testing.assert_allclose(robot.geom_pos, original.geom_pos, atol=1e-9)
    np.testing.assert_allclose(robot.geom_quat, original.geom_quat, atol=1e-9)
    np.testing.assert_array_equal(robot.geom_contype, original.geom_contype)
    np.testing.assert_array_equal(robot.geom_conaffinity, original.geom_conaffinity)
    require(model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT, "Wrong coupling type")
    require(model.eq_obj1id[0] == model.joint(RIGHT).id
            and model.eq_obj2id[0] == model.joint(LEFT).id, "Wrong coupling joints")
    np.testing.assert_allclose(model.eq_data[0, :5], [0, -1, 0, 0, 0])
    require(not np.any(robot.jnt_type == mujoco.mjtJoint.mjJNT_FREE), "Robot base is floating")
    require(model.opt.timestep == 0.001, "Expected a 1 ms timestep")
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    require(model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            "Expected implicitfast integration")
    require(model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and model.opt.impratio == 10,
            "Expected elliptic contact cones and impratio=10 for grasp stability")
    model.site("grasp_site")
    model.camera("demo_cam")
    for color in COLORS:
        model.body(f"cube_{color}")
        model.site(f"target_{color}")
    for path in (ROBOT, SCENE):
        xml = ET.parse(path).getroot()
        for element in xml.iter():
            for attribute in ("file", "meshdir", "texturedir"):
                value = element.get(attribute)
                require(value is None or not Path(value).is_absolute(),
                        f"Nonportable asset reference in {path.name}: {value}")

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "Nonfinite home state")
    penetrations = [
        f"{model.geom(int(c.geom1)).name} / {model.geom(int(c.geom2)).name}: {c.dist:.6g} m"
        for c in data.contact if c.dist < -1e-5
    ]
    require(not penetrations, "Home pose has penetrating contacts: " + "; ".join(penetrations))
    require(not any(w.number for w in data.warning), "MuJoCo warnings at home")
    print("PASS: SHA-256 matches for the URDF and all 10 meshes")
    print("PASS: joint types/limits, fixed base, source inertials and collision geometry")
    print("PASS: 6 arm DoF + 1 coupled gripper DoF; exactly one right = -left constraint")
    print(f"PASS: scene nq={model.nq}, nv={model.nv}; 3 cubes, 3 targets, table, floor, camera")
    print("PASS: relative asset paths; 1 ms timestep; gravity -9.81; implicitfast")
    print("PASS: finite preview home pose, no penetrating contacts or MuJoCo warnings")
    print("NOTE: preview pauses physics; use tools/check.py hold for controlled gravity holding.")
    return model, data


def snapshot(model: mujoco.MjModel, data: mujoco.MjData, destination: Path) -> None:
    import imageio.v2 as imageio

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0  # URDF collision primitives
    option.geomgroup[1] = 1  # CAD visuals
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera="demo_cam", scene_option=option)
        imageio.imwrite(destination, renderer.render())
    print(f"Saved paused model preview: {destination}")


def preview(model: mujoco.MjModel, data: mujoco.MjData, collisions: bool) -> None:
    import mujoco.viewer

    print("PAUSED MODEL PREVIEW: physics does not advance; the robot is not being held by PD.")
    print("Inspect the arm, fingers, table and cubes. Close the viewer to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = (0.4, 0, 0.3)
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -25
            viewer.opt.geomgroup[0] = int(collisions)
            viewer.opt.geomgroup[1] = int(not collisions)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-from", type=Path, metavar="FOLDER",
                        help="Copy the supplied ForteME.urdf and its meshes into this repo")
    parser.add_argument("--check", action="store_true", help="Validate existing generated files without rebuilding")
    parser.add_argument("--preview", action="store_true", help="Inspect the existing scene with physics paused")
    parser.add_argument("--collision-geoms", action="store_true", help="Show primitives instead of CAD in preview")
    parser.add_argument("--snapshot", type=Path, metavar="PNG", help="Save an offscreen paused-scene image")
    args = parser.parse_args()
    if args.collision_geoms and not args.preview:
        parser.error("--collision-geoms requires --preview")
    if args.import_from and (args.check or args.preview):
        parser.error("--import-from cannot be combined with --check or --preview")
    try:
        if args.import_from:
            import_source(args.import_from)
        if not args.check and not args.preview:
            build()
            print(f"Built {ROBOT}\nBuilt {SCENE}")
        model, data = check()
        if args.snapshot:
            snapshot(model, data, args.snapshot)
        if args.preview:
            preview(model, data, args.collision_geoms)
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
