import warp as wp

import mpm_implicit.renderer as rd
from core.progress import MPMProgress
from mpm_implicit.grid import Grid
from mpm_implicit.particles import Particles
from mpm_implicit.solver import Solver
from mpm_implicit.utils import import_mesh


def main():
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
        import_mesh("assets/models/diamond.obj"),
    ]

    solver = Solver(grid, particles, obstacles)

    rd.init(grid, obstacles)

    with MPMProgress(duration=4.0, fps=60) as progress:
        while solver.t < progress.duration:
            solver.update()
            progress.set_t(solver.t)
            if progress.should_render:
                rd.render(solver.t, solver.particles.positions.numpy())


if __name__ == "__main__":
    main()
