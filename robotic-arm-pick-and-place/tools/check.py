"""One entry point for the preserved model, gripper, hold, and IK diagnostics."""

from pathlib import Path
import runpy
import sys


TOOLS = Path(__file__).resolve().parent
COMMANDS = {"model": TOOLS / "build_forte_scene.py",
            "gripper": TOOLS / "diagnostics" / "gripper.py",
            "hold": TOOLS / "diagnostics" / "hold.py",
            "ik": TOOLS / "diagnostics" / "ik.py"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tools/check.py {model,gripper,hold,ik} [options]")
        print("Example: python tools/check.py ik --viewer --task green")
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown check: {command}; choose {', '.join(COMMANDS)}")
    target = COMMANDS[command]
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
