import sys

import warp as wp

import mpm2d.renderer as rd
from core.progress import MPMProgress
from mpm2d.grid import Grid
from mpm2d.particles import Particles
from mpm2d.solver import Method, Solver
from mpm2d.utils import box_polygon


def main(method: Method = Method.EXPLICIT):
    min_coord = (-2.0, -0.05)
    max_coord = (2.0, 4.0)
    dimensions = (200, 305)

    grid = Grid()
    grid.init(
        min_coord=wp.vec2(*min_coord),
        max_coord=wp.vec2(*max_coord),
        dimensions=wp.vec2ui(
            wp.uint32(dimensions[0]),
            wp.uint32(dimensions[1]),
        ),
    )

    cell_size = tuple(
        (hi - lo) / n
        for hi, lo, n in zip(max_coord, min_coord, dimensions, strict=True)
    )

    particles = Particles()
    particles.sample_circle(
        center=wp.vec2(0.0, 2.5),
        radius=0.2,
        velocity=wp.vec2(-3.0, 0.0),
        particle_diam=cell_size[0] * 0.5,
    )

    obstacles = [
        box_polygon((-2.0, -0.1), (2.0, 0.1)),
        box_polygon((-1.13, 0.0), (-0.93, 4.0)),
    ]

    solver = Solver(grid, particles, obstacles, method=method)

    rd.init(grid, obstacles)

    with MPMProgress(duration=4.0, fps=60) as progress:
        while solver.t < progress.duration:
            solver.update()
            progress.set_t(solver.t)
            if progress.should_render:
                rd.render(solver.t, solver.particles)


if __name__ == "__main__":
    main(Method(sys.argv[1]) if len(sys.argv) > 1 else Method.EXPLICIT)
