import warp as wp

from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles
from mpm_explicit.utils import bspline_w, bspline_dw


@wp.struct
class Weights:
    base: wp.array[wp.vec3i]
    wx: wp.array[wp.vec4]
    wy: wp.array[wp.vec4]
    wz: wp.array[wp.vec4]
    dwx: wp.array[wp.vec4]
    dwy: wp.array[wp.vec4]
    dwz: wp.array[wp.vec4]

    def init(self, n: int):
        self.base = wp.empty(n, dtype=wp.vec3i, device="cuda")
        self.wx = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.wy = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.wz = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.dwx = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.dwy = wp.empty(n, dtype=wp.vec4, device="cuda")
        self.dwz = wp.empty(n, dtype=wp.vec4, device="cuda")


class Solver:
    grid: Grid
    particles: Particles

    dt: float
    t: float = 0

    _weights: Weights

    def __init__(self, grid: Grid, particles: Particles, dt: float = 0.001):
        self.grid = grid
        self.particles = particles
        self.dt = dt

        self._weights = Weights()
        self._weights.init(len(particles))

    def update(self):
        self.p2g()
        self.update_nodes()
        self.g2p()
        self.update_particles()
        self.t += self.dt

    def p2g(self):
        wp.launch(
            kernel=k_compute_weights,
            dim=len(self.particles),
            inputs=[self.particles, self.grid, self._weights]
        )

        wp.launch(
            kernel=k_p2g,
            dim=len(self.particles),
            inputs=[self.particles, self.grid, self._weights]
        )

        wp.launch(
            kernel=k_normalize_grid_velocity,
            dim=self.grid.dimensions,
            inputs=[self.grid]
        )

    def update_nodes(self):
        if self.t == 0.0:
            initial_densities: wp.array[float] = wp.zeros(shape=len(self.particles), dtype=float, device="cuda")

            wp.launch(
                kernel=k_calculate_initial_density,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, initial_densities]
            )

            wp.launch(
                kernel=k_set_initial_volumes,
                dim=len(self.particles),
                inputs=[self.particles, initial_densities]
            )

        pass

    def g2p(self):
        pass

    def update_particles(self):
        pass


@wp.kernel
def k_compute_weights(particles: Particles, grid: Grid, weights: Weights):
    p = wp.tid()

    position = particles.positions[p]
    rel_pos = position - grid.min_coord

    base_i = wp.int32(wp.floor(rel_pos[0] / grid.cell_size[0])) - 1
    base_j = wp.int32(wp.floor(rel_pos[1] / grid.cell_size[1])) - 1
    base_k = wp.int32(wp.floor(rel_pos[2] / grid.cell_size[2])) - 1

    weights.base[p] = wp.vec3i(base_i, base_j, base_k)

    wx = wp.vec4(0.0)
    wy = wp.vec4(0.0)
    wz = wp.vec4(0.0)
    dwx = wp.vec4(0.0)
    dwy = wp.vec4(0.0)
    dwz = wp.vec4(0.0)

    for d in range(4):
        node_x = grid.min_coord[0] + float(base_i + d) * grid.cell_size[0]
        node_y = grid.min_coord[1] + float(base_j + d) * grid.cell_size[1]
        node_z = grid.min_coord[2] + float(base_k + d) * grid.cell_size[2]

        dx = (position[0] - node_x) / grid.cell_size[0]
        dy = (position[1] - node_y) / grid.cell_size[1]
        dz = (position[2] - node_z) / grid.cell_size[2]

        wx[d] = bspline_w(dx)
        wy[d] = bspline_w(dy)
        wz[d] = bspline_w(dz)

        dwx[d] = bspline_dw(dx) / grid.cell_size[0]
        dwy[d] = bspline_dw(dy) / grid.cell_size[1]
        dwz[d] = bspline_dw(dz) / grid.cell_size[2]

    weights.wx[p] = wx
    weights.wy[p] = wy
    weights.wz[p] = wz
    weights.dwx[p] = dwx
    weights.dwy[p] = dwy
    weights.dwz[p] = dwz


@wp.kernel
def k_p2g(particles: Particles, grid: Grid, weights: Weights):
    p = wp.tid()

    mass = particles.masses[p]
    velocity = particles.velocities[p]

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if i < 0 or i >= grid.masses.shape[0] or \
                        j < 0 or j >= grid.masses.shape[1] or \
                        k < 0 or k >= grid.masses.shape[2]:
                    continue

                weight = wx[di] * wy[dj] * wz[dk]

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


@wp.kernel
def k_calculate_initial_density(grid: Grid, weights: Weights, initial_densities: wp.array[float]):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    density = 0.0

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if i < 0 or i >= grid.masses.shape[0] or \
                        j < 0 or j >= grid.masses.shape[1] or \
                        k < 0 or k >= grid.masses.shape[2]:
                    continue

                weight = wx[di] * wy[dj] * wz[dk]
                wp.atomic_add(
                    initial_densities,
                    p,
                    weight * grid.masses[i, j, k] / (grid.cell_size[0] * grid.cell_size[1] * grid.cell_size[2])
                )


@wp.kernel
def k_set_initial_volumes(particles: Particles, initial_densities: wp.array[float]):
    p = wp.tid()

    density = initial_densities[p]
    volume = particles.masses[p] / density

    particles.volumes[p] = volume
