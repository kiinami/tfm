import warp as wp

import mpm_explicit.renderer as rd
from core.progress import MPMProgress
from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles, sphere_particle_count
from mpm_explicit.solver import Solver
from mpm_explicit.utils import import_mesh


def main():
    max_coord = (1.0, 2.0, 2.0)
    min_coord = (-1.0, 0.0, 0.0)
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

    particles = Particles()

    cell_size = tuple(
        (hi - lo) / n
        for hi, lo, n in zip(max_coord, min_coord, dimensions, strict=True)
    )
    particle_diam = cell_size[0] * 0.5
    emitters = [
        {
            "center": wp.vec3(0.0, 1.7, 1.7),
            "radius": 0.15,
            "velocity": wp.vec3(0.0, -2.0, -2.0),
        },
        {
            "center": wp.vec3(0.0, 1.7, 0.3),
            "radius": 0.2,
            "velocity": wp.vec3(0.0, -3.0, 3.0),
        },
        {
            "center": wp.vec3(0.0, 0.37, 1.75),
            "radius": 0.2,
            "velocity": wp.vec3(0.0, 3.0, -2.0),
        },
    ]

    total_particles = sum(
        sphere_particle_count(e["radius"], particle_diam) for e in emitters
    )
    particles.init(total_particles)

    offset = 0
    for emitter in emitters:
        offset += particles.fill_sphere(
            offset=offset,
            center=emitter["center"],
            radius=emitter["radius"],
            particle_diam=particle_diam,
            velocity=emitter["velocity"],
        )

    particles.fill_deformations()

    obstacles = [
        import_mesh("assets/models/floor_thick.obj"),
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
