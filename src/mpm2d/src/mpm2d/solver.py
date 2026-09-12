import warp as wp

from mpm2d.constants import DEFAULT_DT, EPSILON, GRAVITY, COULOMB_FRICTION, BOUNDARY_PADDING_CELLS, PICFLIP_ALPHA
from mpm2d.grid import Grid, grid_index_to_coord
from mpm2d.particles import Particles, mu_, lambda_
from mpm2d.utils import bspline_w, bspline_dw, extract_rotation, cofactor, safe_svd2


@wp.struct
class Weights:
    base: wp.array[wp.vec2i]
    wx: wp.array[wp.vec4]
    wy: wp.array[wp.vec4]
    dwx: wp.array[wp.vec4]
    dwy: wp.array[wp.vec4]

    def init(self, n: int):
        self.base = wp.empty(shape=n, dtype=wp.vec2i)
        self.wx = wp.empty(shape=n, dtype=wp.vec4)
        self.wy = wp.empty(shape=n, dtype=wp.vec4)
        self.dwx = wp.empty(shape=n, dtype=wp.vec4)
        self.dwy = wp.empty(shape=n, dtype=wp.vec4)


class Solver:
    grid: Grid
    particles: Particles

    dt: float
    t: float = 0

    _weights: Weights
    _initial_densities: wp.array[float]

    _active_flags: wp.array[int]
    _active_offsets: wp.array[int]
    _active_nodes: wp.array[wp.vec2i]
    _active_count: wp.array[wp.uint32]

    _first: wp.array[int]
    _graph: wp.Graph

    def __init__(
            self,
            grid: Grid,
            particles: Particles,
            dt: float = DEFAULT_DT,
    ):
        self.grid = grid
        self.particles = particles
        self.dt = dt

        self._weights = Weights()
        self._weights.init(len(particles))
        self._first = wp.full(shape=[1], dtype=int, value=1)
        self._initial_densities = wp.zeros(shape=len(self.particles), dtype=float)

        flat_size = self.grid.flat_dimensions
        self._active_flags = wp.zeros(shape=[flat_size], dtype=int)
        self._active_offsets = wp.zeros(shape=[flat_size], dtype=int)
        self._active_nodes = wp.zeros(shape=[flat_size], dtype=wp.vec2i)
        self._active_count = wp.zeros(shape=[1], dtype=wp.uint32)

        self._graph = self._capture_graph()

    def _capture_graph(self):
        with wp.ScopedCapture(capture_mode=wp.CaptureMode.RELAXED) as first_capture:
            wp.launch(
                kernel=k_solver_calculate_initial_densities,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, self._initial_densities],
            )

            wp.launch(
                kernel=k_solver_set_initial_volumes,
                dim=len(self.particles),
                inputs=[self.particles, self._initial_densities]
            )

            self._first.fill_(0)

        with wp.ScopedCapture(capture_mode=wp.CaptureMode.RELAXED) as capture:
            # p2g
            wp.launch(
                kernel=k_solver_compute_weights,
                dim=self.particles.positions.shape[0],
                inputs=[self.particles, self.grid, self._weights],
            )

            wp.launch(
                kernel=k_solver_p2g,
                dim=self.particles.positions.shape[0],
                inputs=[self.particles, self.grid, self._weights],
            )

            wp.launch(
                kernel=k_solver_normalize_grid,
                dim=self.grid.dimensions,
                inputs=[self.grid, self._active_flags],
            )

            wp.utils.array_scan(
                self._active_flags, self._active_offsets, inclusive=False
            )

            wp.launch(
                kernel=k_solver_compute_active_nodes,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self.grid.flat_dimensions,
                    self._active_flags,
                    self._active_offsets,
                    self._active_nodes,
                    self._active_count,
                ],
            )

            # update nodes
            wp.capture_if(self._first, first_capture.graph)

            wp.launch(
                kernel=k_solver_calculate_forces,
                dim=len(self.particles),
                inputs=[self.grid, self.particles, self._weights]
            )

            wp.launch(
                kernel=k_solver_update_grid,
                dim=self.grid.flat_dimensions,
                inputs=[self.grid, self._active_nodes, self._active_count, self.dt]
            )

            wp.launch(
                kernel=k_solver_grid_collisions,
                dim=self.grid.flat_dimensions,
                inputs=[self.grid, self._active_nodes, self._active_count, self.dt]
            )

            wp.launch(
                kernel=k_solver_update_deformations,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights, self.dt]
            )

            # g2p
            wp.launch(
                kernel=k_solver_g2p,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights]
            )

            # update particles
            wp.launch(
                kernel=k_solver_particle_collisions,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self.dt]
            )

            wp.launch(
                kernel=k_solver_advect,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self.dt]
            )

            # clear
            self.grid.clear()

        return capture.graph

    def update(self):
        wp.capture_launch(self._graph)
        self.t += self.dt


@wp.kernel
def k_solver_compute_weights(particles: Particles, grid: Grid, weights: Weights):
    p = wp.tid()

    position = particles.positions[p]
    rel_pos = position - grid.min_coord

    base_i = wp.int32(wp.floor(rel_pos[0] / grid.cell_size[0])) - 1
    base_j = wp.int32(wp.floor(rel_pos[1] / grid.cell_size[1])) - 1

    weights.base[p] = wp.vec2i(base_i, base_j)

    wx = wp.vec4(0.0)
    wy = wp.vec4(0.0)
    dwx = wp.vec4(0.0)
    dwy = wp.vec4(0.0)

    for d in range(4):
        node_x = grid.min_coord[0] + float(base_i + d) * grid.cell_size[0]
        node_y = grid.min_coord[1] + float(base_j + d) * grid.cell_size[1]

        dx = (position[0] - node_x) / grid.cell_size[0]
        dy = (position[1] - node_y) / grid.cell_size[1]

        wx[d] = bspline_w(dx)
        wy[d] = bspline_w(dy)

        dwx[d] = bspline_dw(dx) / grid.cell_size[0]
        dwy[d] = bspline_dw(dy) / grid.cell_size[1]

    weights.wx[p] = wx
    weights.wy[p] = wy
    weights.dwx[p] = dwx
    weights.dwy[p] = dwy


@wp.kernel
def k_solver_p2g(particles: Particles, grid: Grid, weights: Weights):
    p = wp.tid()

    mass = particles.masses[p]
    velocity = particles.velocities[p]

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]

    for dj in range(4):
        for di in range(4):
            i = base[0] + di
            j = base[1] + dj

            if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
            ):
                continue

            weight = wx[di] * wy[dj]

            wp.atomic_add(grid.masses, i, j, mass * weight)
            wp.atomic_add(grid.velocities, i, j, weight * mass * velocity)


@wp.kernel
def k_solver_normalize_grid(grid: Grid, active_flags: wp.array[int]):
    i, j = wp.tid()

    mass = grid.masses[i, j]
    flat_idx = i * wp.int32(grid.dimensions[1]) + j

    if mass > EPSILON:
        grid.velocities[i, j] = grid.velocities[i, j] / mass
        active_flags[flat_idx] = 1
    else:
        grid.velocities[i, j] = wp.vec2(0.0)
        active_flags[flat_idx] = 0


@wp.kernel
def k_solver_compute_active_nodes(
        grid: Grid,
        total_elements: int,
        active_flags: wp.array[int],
        active_offsets: wp.array[int],
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[wp.uint32],
):
    flat_idx = wp.tid()

    if active_flags[flat_idx] == 1:
        offset = active_offsets[flat_idx]

        i = flat_idx // wp.int32(grid.dimensions[1])
        j = flat_idx % wp.int32(grid.dimensions[1])

        active_nodes[offset] = wp.vec2i(i, j)

    if flat_idx == total_elements - 1:
        active_count[0] = wp.uint32(active_offsets[flat_idx] + active_flags[flat_idx])


@wp.kernel
def k_solver_calculate_initial_densities(
        grid: Grid, weights: Weights, initial_densities: wp.array[float]
):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]

    for dj in range(4):
        for di in range(4):
            i = base[0] + di
            j = base[1] + dj

            if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
            ):
                continue

            weight = wx[di] * wy[dj]

            initial_densities[p] += weight * grid.masses[i, j] / (grid.cell_size[0] * grid.cell_size[1])


@wp.kernel
def k_solver_set_initial_volumes(particles: Particles, initial_densities: wp.array[float]):
    p = wp.tid()

    density = initial_densities[p]
    particles.volumes[p] = particles.masses[p] / density


@wp.kernel
def k_solver_calculate_forces(
        grid: Grid, particles: Particles, weights: Weights
):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    dwx = weights.dwx[p]
    dwy = weights.dwy[p]

    F_Ep = particles.elastic_deformations[p]
    F_Pp = particles.plastic_deformations[p]
    xi = particles.hardening_coefs[p]
    mu_0 = particles.mus[p]
    lambda_0 = particles.lambdas[p]

    mu = mu_(F_Pp, xi, mu_0)
    lmbd = lambda_(F_Pp, xi, lambda_0)

    R_Ep = extract_rotation(F_Ep)
    J_Ep = wp.determinant(F_Ep)

    V_0 = particles.volumes[p]

    pk_stress = 2.0 * mu * (F_Ep - R_Ep) + lmbd * (J_Ep - 1.0) * cofactor(F_Ep)
    force = -1.0 * (V_0 * pk_stress @ wp.transpose(F_Ep))

    for dj in range(4):
        for di in range(4):
            i = base[0] + di
            j = base[1] + dj

            if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
            ):
                continue

            grad_weight = wp.vec2(dwx[di] * wy[dj], wx[di] * dwy[dj])

            wp.atomic_add(grid.forces, i, j, force @ grad_weight)


@wp.kernel
def k_solver_update_grid(
        grid: Grid, active_nodes: wp.array[wp.vec2i], active_count: wp.array[wp.uint32], dt: float
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j = coord[0], coord[1]

    mass = grid.masses[i, j]
    vel = grid.velocities[i, j]

    # Elastic force update
    vel += dt * grid.forces[i, j] / mass

    # Gravity update
    vel += dt * GRAVITY

    grid.new_velocities[i, j] = vel


@wp.func
def resolve_wall(
        velocity: wp.vec2,
        test_position: wp.vec2,
        plane_point: wp.vec2,
        normal: wp.vec2,
) -> wp.vec2:
    dist = wp.dot(test_position - plane_point, normal)
    if dist > 0.0:
        return velocity

    velocity_normal = wp.dot(velocity, normal)

    if velocity_normal > 0.0:
        return velocity

    velocity_tangent = velocity - velocity_normal * normal
    tangent_len = wp.length(velocity_tangent)

    if tangent_len <= -COULOMB_FRICTION * velocity_normal:
        return wp.vec2(0.0, 0.0)

    return (
            velocity_tangent
            + COULOMB_FRICTION * velocity_normal * (velocity_tangent / tangent_len)
    )


@wp.kernel
def k_solver_grid_collisions(
        grid: Grid,
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[wp.uint32],
        dt: float,
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j = coord[0], coord[1]

    position = grid_index_to_coord(grid, i, j)
    velocity = grid.new_velocities[i, j]

    nx, ny = grid.dimensions[0], grid.dimensions[1]

    padding = BOUNDARY_PADDING_CELLS * grid.cell_size
    lower = grid_index_to_coord(grid, 0, 0) + padding
    upper = grid_index_to_coord(grid, wp.int32(nx) - 1, wp.int32(ny) - 1) - padding

    # -x wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, lower, wp.vec2(1.0, 0.0))

    # -y wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, lower, wp.vec2(0.0, 1.0))

    # +x wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, upper, wp.vec2(-1.0, 0.0))

    # +y wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, upper, wp.vec2(0.0, -1.0))

    grid.new_velocities[i, j] = velocity


@wp.kernel
def k_solver_update_deformations(
        particles: Particles, grid: Grid, weights: Weights, dt: float
):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]
    dwx = weights.dwx[p]
    dwy = weights.dwy[p]

    F_Ep = particles.elastic_deformations[p]
    F_Pp = particles.plastic_deformations[p]

    velocity_gradient = wp.mat22(0.0)

    for dj in range(4):
        for di in range(4):
            i = base[0] + di
            j = base[1] + dj

            if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
            ):
                continue

            grad_weight = wp.vec2(dwx[di] * wy[dj], wx[di] * dwy[dj])
            velocity = grid.new_velocities[i, j]

            velocity_gradient += wp.outer(velocity, grad_weight)

    I = wp.identity(2, dtype=float)
    F_Ep_tentative = (I + dt * velocity_gradient) @ F_Ep
    F_Pp_tentative = F_Pp
    F_p = F_Ep_tentative @ F_Pp_tentative

    U_p, sigma_p, V_p = safe_svd2(F_Ep_tentative)

    sigma_p_clamped = wp.vec2(0.0)
    low = 1.0 - particles.critical_compressions[p]
    high = 1.0 + particles.critical_stretches[p]
    for i in range(2):
        sigma_p_clamped[i] = wp.clamp(sigma_p[i], low, high)

    sigma_p_clamped_diag = wp.diag(sigma_p_clamped)

    inv_sigma_p_clamped = wp.vec2(
        1.0 / sigma_p_clamped[0],
        1.0 / sigma_p_clamped[1]
    )
    inv_sigma_p_clamped_diag = wp.diag(inv_sigma_p_clamped)

    particles.elastic_deformations[p] = U_p @ sigma_p_clamped_diag @ wp.transpose(V_p)
    particles.plastic_deformations[p] = V_p @ inv_sigma_p_clamped_diag @ wp.transpose(U_p) @ F_p


@wp.kernel
def k_solver_g2p(
        particles: Particles, grid: Grid, weights: Weights
):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]

    v_pic = wp.vec2(0.0)
    v_flip = particles.velocities[p]

    for dj in range(4):
        for di in range(4):
            i = base[0] + di
            j = base[1] + dj

            if (
                    i < 0
                    or i >= grid.masses.shape[0]
                    or j < 0
                    or j >= grid.masses.shape[1]
            ):
                continue

            weight = wx[di] * wy[dj]
            node_velocity = grid.velocities[i, j]
            node_new_velocity = grid.new_velocities[i, j]

            v_pic += weight * node_new_velocity
            v_flip += weight * (node_new_velocity - node_velocity)

    particles.velocities[p] = (1.0 - PICFLIP_ALPHA) * v_pic + PICFLIP_ALPHA * v_flip


@wp.kernel
def k_solver_particle_collisions(particles: Particles, grid: Grid, dt: float):
    p = wp.tid()

    position = particles.positions[p]
    velocity = particles.velocities[p]

    nx, ny = grid.dimensions[0], grid.dimensions[1]

    padding = BOUNDARY_PADDING_CELLS * grid.cell_size
    lower = grid_index_to_coord(grid, 0, 0) + padding
    upper = grid_index_to_coord(grid, wp.int32(nx) - 1, wp.int32(ny) - 1) - padding

    # -x wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, lower, wp.vec2(1.0, 0.0))

    # -y wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, lower, wp.vec2(0.0, 1.0))

    # +x wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, upper, wp.vec2(-1.0, 0.0))

    # +y wall
    test_position = position + dt * velocity
    velocity = resolve_wall(velocity, test_position, upper, wp.vec2(0.0, -1.0))

    particles.velocities[p] = velocity


@wp.kernel
def k_solver_advect(
        particles: Particles, grid: Grid, dt: float
):
    p = wp.tid()
    particles.positions[p] += dt * particles.velocities[p]
