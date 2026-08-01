import numpy as np
import warp as wp

from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles
from mpm_explicit.utils import bspline_w


class Solver:
    grid: Grid
    particles: Particles

    dt: float


    def __init__(self, grid: Grid, particles: Particles, dt: float = 0.001):
        self.grid = grid
        self.particles = particles
        self.dt = dt

    def update(self):
        self.p2g()
        self.update_nodes()
        self.g2p()
        self.update_particles()

    def p2g(self):
        wp.launch(
            kernel=k_p2g,
            dim=len(self.particles),
            inputs=[self.particles, self.grid]
        )

        wp.launch(
            kernel=k_normalize_grid_velocity,
            dim=self.grid.dimensions,
            inputs=[self.grid]
        )

    def update_nodes(self):
        pass

    def g2p(self):
        pass

    def update_particles(self):
        pass

    @property
    def positions(self) -> np.ndarray:
        return


@wp.kernel
def k_p2g(particles: Particles, grid: Grid):
    p = wp.tid()

    position = particles.positions[p]
    mass = particles.masses[p]
    velocity = particles.velocities[p]

    rel_pos = position - grid.min_coord

    base_i = int(wp.floor(rel_pos[0] / grid.cell_size[0])) - 1
    base_j = int(wp.floor(rel_pos[1] / grid.cell_size[1])) - 1
    base_k = int(wp.floor(rel_pos[2] / grid.cell_size[2])) - 1

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base_i + di
                j = base_j + dj
                k = base_k + dk

                if i < 0 or i >= grid.masses.shape[0] or \
                        j < 0 or j >= grid.masses.shape[1] or \
                        k < 0 or k >= grid.masses.shape[2]:
                    continue

                node_position = grid.positions[i, j, k]

                dx = (position[0] - node_position[0]) / grid.cell_size[0]
                dy = (position[1] - node_position[1]) / grid.cell_size[1]
                dz = (position[2] - node_position[2]) / grid.cell_size[2]

                wx = bspline_w(dx)
                wy = bspline_w(dy)
                wz = bspline_w(dz)
                weight = wx * wy * wz

                wp.atomic_add(grid.masses, i, j, k, weight * mass)
                wp.atomic_add(grid.velocities, i, j, k, weight * mass * velocity)


@wp.kernel
def k_normalize_grid_velocity(grid: Grid):
    i, j, k = wp.tid()

    mass = grid.masses[i, j, k]

    if mass > 0.0:
        grid.velocities[i, j, k] = grid.velocities[i, j, k] / mass
        grid.active[i, j, k] = True
    else:
        grid.velocities[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        grid.active[i, j, k] = False
