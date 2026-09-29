import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mpm2d.grid import Grid
from mpm2d.particles import Particles
from mpm2d.utils import Polygon

# rerun's 2D views have the y axis pointing down
_FLIP = np.array([1.0, -1.0], dtype=np.float32)


def init(grid: Grid, obstacles: list[Polygon]):
    rr.init("mpm", spawn=True)

    min_pos = np.array([float(grid.min_coord[0]), float(grid.min_coord[1])])
    max_pos = np.array([float(grid.max_coord[0]), float(grid.max_coord[1])])

    lower_f, upper_f = min_pos * _FLIP, max_pos * _FLIP
    mins = np.minimum(lower_f, upper_f)

    rr.log(
        "mpm/static/box",
        rr.Boxes2D(
            mins=[mins],
            sizes=[np.abs(upper_f - lower_f)],
            colors=[128, 128, 128]
        ),
        static=True
    )

    for i, polygon in enumerate(obstacles):
        points = polygon.points.numpy()
        closed = np.concatenate([points, points[:1]]) * _FLIP
        rr.log(
            f"mpm/static/obstacle_{i}",
            rr.LineStrips2D([closed], colors=[64, 83, 109]),
            static=True
        )

    rr.send_blueprint(
        rrb.Spatial2DView(
            origin="/mpm",
            name="MPM",
            background=[16, 16, 16],
            visual_bounds=rrb.VisualBounds2D(
                x_range=[min_pos[0], max_pos[0]],
                y_range=[-max_pos[1], -min_pos[1]],
            ),
        )
    )


def render(t: float, particles: Particles, contrast: float = 0.5):
    rr.set_time("step", timestamp=t)

    densities = particles.densities.numpy()
    reference_densities = particles.masses.numpy() / particles.volumes.numpy()

    shade = densities / reference_densities * contrast + (1.0 - contrast)
    shade = np.clip(shade, 0.0, 1.0)

    colors = np.repeat((shade * 255.0).astype(np.uint8)[:, None], 3, axis=1)

    rr.log(
        "mpm/particles",
        rr.Points2D(
            positions=particles.positions.numpy() * _FLIP,
            colors=colors,
            radii=rr.Radius.ui_points(2.0),
        ),
    )
