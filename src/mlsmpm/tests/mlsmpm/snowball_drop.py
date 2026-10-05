import sys

import warp as wp

import mlsmpm.renderer as rd
from core.progress import MPMProgress
from mlsmpm.grid import Grid
from mlsmpm.particles import Particles
from mlsmpm.solver import Method, Solver
from mlsmpm.utils import import_mesh


def main(method: Method = Method.EXPLICIT):
    max_coord = (1.0, 1.0, 3.0)
    min_coord = (-1.0, -1.0, 0.0)
    dimensions = (128, 128, 128)

    grid = Grid()
    grid.init(
        min_coord=wp.vec3(*min_coord),
        max_coord=wp.vec3(*max_coord),
        dimensions=wp.vec3ui(
            wp.uint32(dimensions[0]),
            wp.uint32(dimensions[1]),
            wp.uint32(dimensions[2]),
        ),
    )

    cell_size = tuple(
        (hi - lo) / n
        for hi, lo, n in zip(max_coord, min_coord, dimensions, strict=True)
    )

    particles = Particles()
    particles.sample_sphere(
        center=wp.vec3(0.0, 0.0, 2.7),
        radius=0.2,
        velocity=wp.vec3(0.0, 0.0, -3.0),
        particle_diam=cell_size[0] * 0.5,
    )

    obstacles = [
        import_mesh("assets/models/floor_thick.obj"),
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
