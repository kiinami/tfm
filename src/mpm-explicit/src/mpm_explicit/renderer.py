import numpy as np
import rerun as rr

from mpm_explicit.grid import Grid


def init(grid: Grid):
    rr.init("mpm", spawn=True)
    rr.log(
        "world/box",
        rr.Boxes3D(
            centers=[grid.center()],
            half_sizes=[grid.half_dimensions()]
        ),
        static=True
    )


def render(step: int, positions: np.ndarray):
    rr.set_time("step", sequence=step)
    rr.log("world/particles", rr.Points3D(positions=positions))
