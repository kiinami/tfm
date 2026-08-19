import warp as wp
from nvtx import nvtx
from warp.examples.fem.example_burgers import velocity_norm

from mpm_explicit.boundaries import PlaneBoundary, signed_distance
from mpm_explicit.constants import COULOMB_FRICTION, GRAVITY, PICFLIP_ALPHA, MASS_EPSILON, DEFAULT_DT
from mpm_explicit.grid import Grid, flatten, unflatten
from mpm_explicit.particles import Particles, lambda_, mu_
from mpm_explicit.utils import bspline_dw, bspline_w, extract_rotation


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
    boundary: PlaneBoundary

    dt: float
    t: float = 0

    _weights: Weights
    _first: bool

    def __init__(self, grid: Grid, particles: Particles, boundary: PlaneBoundary, dt: float = DEFAULT_DT):
        self.grid = grid
        self.particles = particles
        self.boundary = boundary
        self.dt = dt

        self._weights = Weights()
        self._weights.init(len(particles))
        self._first = True

    def update(self):
        with wp.ScopedTimer(f"Step at t={self.t:05}", synchronize=True, use_nvtx=True):
            self.p2g()
            self.update_nodes()
            self.g2p()
            self.update_particles()
            self.clear()

    def advance(self):
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
            kernel=k_normalize_grid,
            dim=self.grid.dimensions,
            inputs=[self.grid]
        )

        wp.utils.array_scan(self.grid.is_active, self.grid.offsets, inclusive=False)

        wp.launch(
            kernel=k_compact_active_nodes,
            dim=self.grid.flat_dimensions,
            inputs=[self.grid]
        )

    def update_nodes(self):
        if self._first:
            self._first = False
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

        wp.launch(
            kernel=k_calculate_forces,
            dim=len(self.particles),
            inputs=[self.grid, self._weights, self.particles]
        )

        wp.launch(
            kernel=k_update_grid,
            dim=self.grid.active_node_count,
            inputs=[self.grid, self.dt]
        )

        wp.launch(
            kernel=k_calculate_grid_collisions,
            dim=self.grid.active_node_count,
            inputs=[self.grid, self.boundary]
        )

        wp.launch(
            kernel=k_update_deformations,
            dim=len(self.particles),
            inputs=[self.particles, self.grid, self._weights, self.dt]
        )

    def g2p(self):
        wp.launch(
            kernel=k_g2p,
            dim=len(self.particles),
            inputs=[self.particles, self.grid, self._weights, self.dt]
        )

    def update_particles(self):
        wp.launch(
            kernel=k_calculate_particle_collisions,
            dim=len(self.particles),
            inputs=[self.particles, self.boundary]
        )
        wp.launch(
            kernel=k_advect_particle_positions,
            dim=len(self.particles),
            inputs=[self.particles, self.dt]
        )

    def clear(self):
        self.grid.clear()


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

                grid.masses[i, j, k] += weight * mass
                grid.velocities[i, j, k] += weight * mass * velocity


@wp.kernel
def k_normalize_grid(grid: Grid):
    i, j, k = wp.tid()

    mass = grid.masses[i, j, k]

    if mass > MASS_EPSILON:
        grid.velocities[i, j, k] = grid.velocities[i, j, k] / mass
        grid.is_active[flatten(grid.dimensions, i, j, k)] = 1
    else:
        grid.velocities[i, j, k] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_compact_active_nodes(grid: Grid):
    idx = wp.tid()

    if grid.is_active[idx]:
        slot = grid.offsets[idx]
        grid.active_nodes[slot] = idx


@wp.kernel
def k_calculate_initial_density(grid: Grid, weights: Weights, initial_densities: wp.array[float]):
    p = wp.tid()

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
                initial_densities[p] += weight * grid.masses[i, j, k] / (grid.cell_size[0] * grid.cell_size[1] * grid.cell_size[2])


@wp.kernel
def k_set_initial_volumes(particles: Particles, initial_densities: wp.array[float]):
    p = wp.tid()

    density = initial_densities[p]
    volume = particles.masses[p] / density

    particles.volumes[p] = volume


@wp.kernel
def k_calculate_forces(grid: Grid, weights: Weights, particles: Particles):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    dwx = weights.dwx[p]
    dwy = weights.dwy[p]
    dwz = weights.dwz[p]

    F_Ep = particles.elastic_deformations[p]
    F_Pp = particles.plastic_deformations[p]
    xi = particles.hardening_coefs[p]
    mu_0 = particles.mus[p]
    mu = mu_(F_Pp, xi, mu_0)
    lambda_0 = particles.lambdas[p]
    lmbd = lambda_(F_Pp, xi, lambda_0)

    R_Ep = extract_rotation(F_Ep)
    J_Ep = wp.determinant(F_Ep)

    V_0 = particles.volumes[p]

    pk_stress = 2.0 * mu * (F_Ep - R_Ep) + lmbd * (J_Ep - 1.0) * J_Ep * wp.inverse(wp.transpose(F_Ep))
    force = -1.0 * (V_0 * pk_stress @ wp.transpose(F_Ep))

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

                grad_weight = wp.vec3(
                    dwx[di] * wy[dj] * wz[dk],
                    wx[di] * dwy[dj] * wz[dk],
                    wx[di] * wy[dj] * dwz[dk],
                )

                grid.forces[i, j, k] += force @ grad_weight


@wp.kernel
def k_update_grid(grid: Grid, dt: float):
    idx = wp.tid()
    node = unflatten(grid.dimensions, idx)
    i = node[0]
    j = node[1]
    k = node[2]

    mass = grid.masses[i, j, k]
    if mass <= 0.0:
        return

    # existing momentum -> velocity conversion
    vel = grid.velocities[i, j, k]

    # elastic force
    vel += dt * grid.forces[i, j, k] / mass

    # gravity
    vel += dt * GRAVITY

    grid.new_velocities[i, j, k] = vel


@wp.kernel
def k_calculate_grid_collisions(grid: Grid, boundary: PlaneBoundary):
    idx = wp.tid()
    node = unflatten(grid.dimensions, idx)
    i = node[0]
    j = node[1]
    k = node[2]

    position = grid.positions[i, j, k]
    velocity = grid.new_velocities[i, j, k]

    dist = signed_distance(boundary, position)

    if dist <= 0.0:
        normal = boundary.normal
        velocity_normal = wp.dot(velocity, normal)
        velocity_tangent = velocity - velocity_normal * normal

        if velocity_normal > 0.0:
            return

        if wp.length(velocity_tangent) <= -COULOMB_FRICTION * velocity_normal:
            grid.new_velocities[i, j, k] = wp.vec3(0.0)
        else:
            grid.new_velocities[i, j, k] =  velocity_tangent + COULOMB_FRICTION * velocity_normal * (velocity_tangent / wp.length(velocity_tangent))


@wp.kernel
def k_update_deformations(particles: Particles, grid: Grid, weights: Weights, dt: float):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    dwx = weights.dwx[p]
    dwy = weights.dwy[p]
    dwz = weights.dwz[p]

    F_Ep = particles.elastic_deformations[p]
    F_Pp = particles.plastic_deformations[p]

    velocity_gradient = wp.mat33(0.0)

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

                grad_weight = wp.vec3(
                    dwx[di] * wy[dj] * wz[dk],
                    wx[di] * dwy[dj] * wz[dk],
                    wx[di] * wy[dj] * dwz[dk],
                )

                new_velocity = grid.new_velocities[i, j, k]

                velocity_gradient += wp.outer(new_velocity, grad_weight)

    I = wp.identity(3, dtype=wp.float32)
    F_Ep_tentative = (I + dt * velocity_gradient) * F_Ep
    F_Pp_tentative = F_Pp
    F_p = F_Ep_tentative * F_Pp_tentative

    U_p = wp.mat33(1.0)
    V_p = wp.mat33(1.0)
    sigma_p = wp.vec3(0.0)

    wp.svd3(F_Ep_tentative, U_p, sigma_p, V_p)

    sigma_p_clamped = wp.vec3(0.0)
    low = 1.0 - particles.critical_compressions[p]
    high = 1.0 + particles.critical_stretches[p]
    for i in range(3):
        sigma_p_clamped[i] = wp.clamp(sigma_p[i], low, high)

    sigma_p_clamped_diag = wp.diag(sigma_p_clamped)

    particles.elastic_deformations[p] = U_p * sigma_p_clamped_diag * wp.transpose(V_p)
    particles.plastic_deformations[p] = V_p * wp.inverse(sigma_p_clamped_diag) * wp.transpose(U_p) * F_p


@wp.kernel
def k_g2p(particles: Particles, grid: Grid, weights: Weights, dt: float):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    wz = weights.wz[p]

    v_pic = wp.vec3(0.0)
    v_flip = particles.velocities[p]

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
                node_velocity = grid.velocities[i, j, k]
                node_new_velocity = grid.new_velocities[i, j, k]

                v_pic += node_new_velocity * weight
                v_flip += (node_new_velocity - node_velocity) * weight

    particles.velocities[p] = (1.0 - PICFLIP_ALPHA) * v_pic + PICFLIP_ALPHA * v_flip


@wp.kernel
def k_calculate_particle_collisions(particles: Particles, boundary: PlaneBoundary):
    p = wp.tid()

    position = particles.positions[p]
    velocity = particles.velocities[p]

    dist = signed_distance(boundary, position)

    if dist <= 0.0:
        normal = boundary.normal
        velocity_normal = wp.dot(velocity, normal)
        velocity_tangent = velocity - velocity_normal * normal

        if velocity_normal > 0.0:
            return

        if wp.length(velocity_tangent) <= -COULOMB_FRICTION * velocity_normal:
            particles.velocities[p] = wp.vec3(0.0)
        else:
            particles.velocities[p] = velocity_tangent + COULOMB_FRICTION * velocity_normal * (velocity_tangent / wp.length(velocity_tangent))


@wp.kernel
def k_advect_particle_positions(particles: Particles, dt: float):
    p = wp.tid()
    particles.positions[p] += dt * particles.velocities[p]

