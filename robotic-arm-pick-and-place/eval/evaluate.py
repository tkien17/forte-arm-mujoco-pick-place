"""
eval/evaluate.py  —  measure pick-and-place reliability over many episodes.

Each episode randomizes the cube positions, then the arm attempts to place all
three on their pads. We record per-color and overall success rates and save a
bar chart to media/success_rates.png.

Run:  python eval/evaluate.py --episodes 20
(No rendering, so plain python is fine and it runs fast.)
"""

import sys, os, argparse
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")           # headless backend (no display needed)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers.scripted_controller import ScriptedController

MODEL_PATH = "assets/franka_emika_panda/scene.xml"
TASKS = [("cube_red", "target_red"),
         ("cube_green", "target_green"),
         ("cube_blue", "target_blue")]


def run(episodes, seed=0):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    ctrl = ScriptedController(model, data, seed=seed)

    counts = {c: 0 for c, _ in TASKS}     # successes per color
    for ep in range(episodes):
        ctrl.reset(randomize=True)
        ctrl._settle(250)
        for cube, target in TASKS:
            if ctrl.pick_and_place(cube, target):
                counts[cube] += 1
        done = sum(counts.values())
        print(f"  episode {ep+1}/{episodes} | running successes: {done}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    counts = run(args.episodes)
    colors = [c.replace("cube_", "") for c, _ in TASKS]
    rates = [counts[c] / args.episodes for c, _ in TASKS]
    overall = sum(counts.values()) / (args.episodes * len(TASKS))

    print("\n==== RESULTS ====")
    for c, r in zip(colors, rates):
        print(f"  {c:6s}: {r*100:5.1f}% success")
    print(f"  OVERALL: {overall*100:.1f}% over {args.episodes} episodes")

    # bar chart
    os.makedirs("media", exist_ok=True)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(colors, [r * 100 for r in rates],
                   color=["#e23b3b", "#36c65a", "#3b6fe2"])
    plt.axhline(overall * 100, ls="--", color="gray",
                label=f"overall {overall*100:.0f}%")
    plt.ylim(0, 105)
    plt.ylabel("Success rate (%)")
    plt.title(f"Pick-and-Place Success over {args.episodes} randomized episodes")
    plt.legend()
    for b, r in zip(bars, rates):
        plt.text(b.get_x() + b.get_width()/2, r*100 + 2, f"{r*100:.0f}%",
                 ha="center")
    plt.tight_layout()
    plt.savefig("media/success_rates.png", dpi=120)
    print("\nSaved chart -> media/success_rates.png")


if __name__ == "__main__":
    main()
