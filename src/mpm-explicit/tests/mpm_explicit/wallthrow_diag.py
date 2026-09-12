import time

import mpm_explicit.renderer as rd
import numpy as np
import warp as wp
from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles
from mpm_explicit.solver import Solver
from mpm_explicit.utils import import_mesh

DURATION = 0.4
DT = 1e-5

PARTICLES_PER_CELL = 16

LOG_EVERY_SECONDS = 0.005
FPS = 30


def main():
    print("Initializing warp and compiling kernels")
    wp.init()

    grid = Grid()
    grid.init(
        min_coord=wp.vec3(-2.0, -2.0, -0.05),
        max_coord=wp.vec3(2.0, 2.0, 4.0),
        dimensions=wp.vec3ui(wp.uint32(200), wp.uint32(200), wp.uint32(305)),
    )

    cell_volume = (1.10 / 110) ** 3
    derived_density = PARTICLES_PER_CELL / cell_volume

    snowball_center = wp.vec3(0.0, 0.0, 2.5)
    snowball_radius = 0.2

    particles = Particles()
    particles.sample_packed_snowball(
        center=snowball_center,
        radius=snowball_radius,
        particle_density=derived_density,
        stiffness_outer_mult=1.5,
    )
    particles.velocities.fill_(wp.vec3(0.0, -14.0, 0.0))

    print(f"n_particles = {len(particles)}")

    obstacles = [
        import_mesh("assets/models/floor.obj"),
        import_mesh("assets/models/wall.obj"),
    ]

    solver = Solver(grid, particles, obstacles, DT)

    rd.init(grid, obstacles)
    rd.render(solver.t, solver.particles.positions.numpy())

    total_steps = int(round(DURATION / DT))
    log_every_steps = max(1, int(round(LOG_EVERY_SECONDS / DT)))
    frame_duration = 1.0 / FPS
    next_frame_time = 0.0

    print(
        "t        | JEp_min  JEp_max | JPp_min  JPp_max  JPp_mean | "
        "clamp_lo% clamp_hi% | strain_max strain_mean | "
        "speed_max speed_mean | bbox_y (min,max) | bbox_z (min,max)"
    )

    t0 = time.time()
    for step in range(total_steps):
        solver.update()

        if solver.t >= next_frame_time:
            rd.render(solver.t, solver.particles.positions.numpy())
            next_frame_time += frame_duration

        if step % log_every_steps == 0:
            f_ep = solver.particles.elastic_deformations.numpy()
            f_pp = solver.particles.plastic_deformations.numpy()
            pos = solver.particles.positions.numpy()
            vel = solver.particles.velocities.numpy()

            j_ep = np.linalg.det(f_ep)
            j_pp = np.linalg.det(f_pp)

            sigma = np.linalg.svd(f_ep, compute_uv=False)
            clamp_lo_frac = float(
                np.mean(np.any(sigma <= 0.975 + 1e-4, axis=1))
            )
            clamp_hi_frac = float(
                np.mean(np.any(sigma >= 1.0075 - 1e-4, axis=1))
            )

            i3 = np.eye(3)
            strain_norm = np.linalg.norm(f_ep - i3, axis=(1, 2))

            speed = np.linalg.norm(vel, axis=1)

            elapsed = time.time() - t0
            print(
                f"{solver.t:8.5f} | "
                f"{j_ep.min():7.4f} {j_ep.max():7.4f} | "
                f"{j_pp.min():7.4f} {j_pp.max():7.4f} {j_pp.mean():7.4f} | "
                f"{clamp_lo_frac * 100:8.3f}% {clamp_hi_frac * 100:8.3f}% | "
                f"{strain_norm.max():10.5f} {strain_norm.mean():11.6f} | "
                f"{speed.max():9.3f} {speed.mean():10.3f} | "
                f"({pos[:, 1].min():6.3f},{pos[:, 1].max():6.3f}) | "
                f"({pos[:, 2].min():6.3f},{pos[:, 2].max():6.3f}) "
                f"[{elapsed:.1f}s wall]"
            )

    print("done")


if __name__ == "__main__":
    main()
