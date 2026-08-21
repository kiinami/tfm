import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import warp as wp

from mpm_explicit.grid import Grid


def init(grid: Grid, obstacles: list[wp.Mesh]):
    rr.init("mpm", spawn=True)

    min_pos = [float(grid.min_coord[0]), float(grid.min_coord[1]), float(grid.min_coord[2])]
    max_pos = [float(grid.max_coord[0]), float(grid.max_coord[1]), float(grid.max_coord[2])]

    center = [
        0.5 * (min_pos[0] + max_pos[0]),
        0.5 * (min_pos[1] + max_pos[1]),
        0.5 * (min_pos[2] + max_pos[2]),
    ]

    half_sizes = [
        0.5 * (max_pos[0] - min_pos[0]),
        0.5 * (max_pos[1] - min_pos[1]),
        0.5 * (max_pos[2] - min_pos[2]),
    ]

    rr.log(
        "mpm/box",
        rr.Boxes3D(
            half_sizes=half_sizes,
            centers=center,
            colors=[128, 128, 128]
        ),
        static=True
    )

    for i, mesh in enumerate(obstacles):
        rr.log(
            f"mpm/obstacle_{i}",
            rr.Mesh3D(
                vertex_positions=mesh.points.numpy(),
                triangle_indices=mesh.indices.numpy().reshape(-1, 3),
                vertex_colors=[128, 128, 128, 128]
            ),
            static=True
        )


def render(t: float, positions: np.ndarray):
    rr.set_time("step", timestamp=t)
    rr.log("world/particles", rr.Points3D(positions=positions, colors=[255, 255, 255]))
