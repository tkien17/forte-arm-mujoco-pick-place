"""
utils/recording.py  —  record a simulation to GIF/MP4 with no display needed.

Uses MuJoCo's offscreen Renderer. On a headless machine set the env var
MUJOCO_GL=osmesa before running. On a normal Mac the default works.

Typical use (see scripts/record_demo.py):
    rec = Recorder(model, camera="demo_cam", every=20)
    ctrl.recorder = rec
    ctrl.pick_and_place(...)
    rec.save_gif("media/demo.gif")
"""

import numpy as np
import mujoco
import imageio.v2 as imageio


class Recorder:
    def __init__(self, model, width=640, height=480, camera=-1, fps=30, every=20):
        # `every`: capture one frame per N physics steps (physics ~500 Hz,
        # so every=20 -> ~25 captured fps, a smooth but small GIF).
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = camera
        self.fps = fps
        self.every = every
        self._count = 0
        self.frames = []

    def capture(self, data):
        self._count += 1
        if self._count % self.every != 0:
            return
        self.renderer.update_scene(data, camera=self.camera)
        self.frames.append(self.renderer.render())

    def save_gif(self, path):
        imageio.mimsave(path, self.frames, fps=self.fps, loop=0)
        print(f"Saved {len(self.frames)} frames -> {path}")

    def save_mp4(self, path):
        imageio.mimsave(path, self.frames, fps=self.fps)
        print(f"Saved {len(self.frames)} frames -> {path}")
