import warp as wp

import mpm_explicit.renderer as rd
from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles
from mpm_explicit.solver import Solver


def main():
    grid = Grid()
    grid.init(
        min_coord=wp.vec3(0, 0, 0),
        max_coord=wp.vec3(100, 100, 100),
        dimensions=wp.vec3ui(wp.uint32(100), wp.uint32(100), wp.uint32(100))
    )

    particles = Particles()
    particles.sample_cube(
        min_coord=wp.vec3(25, 25, 25),
        cell_size=wp.vec3(1, 1, 1),
        dimensions=wp.vec3ui(wp.uint32(50), wp.uint32(50), wp.uint32(50)),
        particles_per_cell=8,
        seed=42
    )

    solver = Solver(grid, particles)

    rd.init(grid)

    for step in range(10):
        solver.update()

        rd.render(solver.t, solver.particles.positions.numpy())


if __name__ == "__main__":
    main()
