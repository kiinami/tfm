import warp as wp
from rich.progress import track

import mpm_explicit.renderer as rd
from mpm_explicit.boundaries import PlaneBoundary
from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles
from mpm_explicit.solver import Solver


DURATION = 5.0
FPS = 30
DT = 0.00001


def main():
    print("Initializing warp and compiling kernels")
    wp.init()

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

    plane = PlaneBoundary()
    plane.point = wp.vec3(0, 0, 0)
    plane.normal = wp.vec3(0, 0, 1)

    solver = Solver(grid, particles, plane, DT)

    rd.init(grid, plane)
    rd.render(solver.t, solver.particles.positions.numpy())

    total_steps = int(round(DURATION / DT))
    frame_duration = 1.0 / FPS
    next_frame_time = 0.0
    for _ in track(range(total_steps), description="Running simulation steps..."):
        solver.update()
        if solver.t >= next_frame_time:
            rd.render(solver.t, solver.particles.positions.numpy())
            next_frame_time += frame_duration
        solver.advance()


if __name__ == "__main__":
    main()
