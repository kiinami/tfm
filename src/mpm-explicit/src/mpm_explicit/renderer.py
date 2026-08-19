import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mpm_explicit.boundaries import PlaneBoundary
from mpm_explicit.grid import Grid


def init(grid: Grid, boundary: PlaneBoundary):
    rr.init("mpm", spawn=True)
    plane_normal, plane_distance = boundary.hesse_normal_form()

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            name="3D Scene",
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True,
                spacing=0.5,
                plane=rr.components.Plane3D(normal=plane_normal, distance=plane_distance),
                color=[128, 128, 128],
                stroke_width=1.0,
            ),
        )
    )
    rr.send_blueprint(blueprint)


def render(t: float, positions: np.ndarray):
    rr.set_time("step", timestamp=t)
    rr.log("world/particles", rr.Points3D(positions=positions, colors=[255, 255, 255]))
