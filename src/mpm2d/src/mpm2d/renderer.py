import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mpm2d.grid import Grid

_FLIP = np.array([1.0, -1.0], dtype=np.float32)


def init(grid: Grid):
    rr.init("mpm", spawn=True)

    min_pos = np.array([float(grid.min_coord[0]), float(grid.min_coord[1])])
    max_pos = np.array([float(grid.max_coord[0]), float(grid.max_coord[1])])

    _log_box("mpm/static/domain", min_pos, max_pos, [80, 80, 80])

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


def _log_box(path: str, lower: np.ndarray, upper: np.ndarray, color):
    lower_f, upper_f = lower * _FLIP, upper * _FLIP
    mins = np.minimum(lower_f, upper_f)
    rr.log(
        path,
        rr.Boxes2D(mins=[mins], sizes=[np.abs(upper_f - lower_f)], colors=[color]),
        static=True,
    )


def render(t: float, positions: np.ndarray):
    rr.set_time("step", timestamp=t)
    rr.log(
        "mpm/particles",
        rr.Points2D(
            positions=positions * _FLIP,
            colors=[255, 255, 255],
            radii=rr.Radius.ui_points(2.0),
        ),
    )
