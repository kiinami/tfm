from functools import cached_property

import warp as wp

from mpm_explicit.constants import COULOMB_FRICTION, GRAVITY, PICFLIP_ALPHA, EPSILON, DEFAULT_DT, \
    MAX_COLLISION_DIST
from mpm_explicit.grid import Grid, grid_index_to_coord
from mpm_explicit.particles import Particles, lambda_, mu_
from mpm_explicit.utils import bspline_dw, bspline_w, extract_rotation, cofactor, safe_svd3


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
    obstacles: list[wp.Mesh]

    dt: float
    t: float = 0

    _weights: Weights
    _initial_densities: wp.array[float]

    _active_flags: wp.array[int]
    _active_offsets: wp.array[int]
    _active_nodes: wp.array[wp.vec3i]
    _active_count: wp.array[int]

    _first: wp.array[int]
    _graph: wp.Graph

    def __init__(self, grid: Grid, particles: Particles, obstacles: list[wp.Mesh], dt: float = DEFAULT_DT):
        self.grid = grid
        self.particles = particles
        self.obstacles = obstacles
        self.dt = dt

        self._weights = Weights()
        self._weights.init(len(particles))
        self._first = wp.zeros(1, dtype=int)
        self._first.fill_(1)
        self._initial_densities = wp.zeros(shape=len(self.particles), dtype=float, device="cuda")

        flat_size = self.grid.flat_dimensions
        self._active_flags = wp.zeros(shape=flat_size, dtype=wp.int32, device="cuda")
        self._active_offsets = wp.zeros(shape=flat_size, dtype=wp.int32, device="cuda")
        self._active_nodes = wp.zeros(shape=flat_size, dtype=wp.vec3i, device="cuda")
        self._active_count = wp.zeros(shape=1, dtype=wp.int32, device="cuda")

        self._graph = self._capture_graph()

    @cached_property
    def obstacle_ids(self) -> wp.array[wp.uint64]:
        return wp.array([obs.id for obs in self.obstacles], dtype=wp.uint64)

    def _capture_graph(self):
        with wp.ScopedCapture(device="cuda", capture_mode=wp.CaptureMode.RELAXED) as first_capture:
            wp.launch(
                kernel=k_calculate_initial_density,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, self._initial_densities]
            )

            wp.launch(
                kernel=k_set_initial_volumes,
                dim=len(self.particles),
                inputs=[self.particles, self._initial_densities]
            )

            self._first.fill_(0)

        with wp.ScopedCapture(device="cuda", capture_mode=wp.CaptureMode.RELAXED) as capture:
            # p2g
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
                inputs=[self.grid, self._active_flags]
            )

            wp.utils.array_scan(self._active_flags, self._active_offsets, inclusive=False)

            wp.launch(
                kernel=k_collect_active_nodes,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self._active_flags,
                    self._active_offsets,
                    self._active_nodes,
                    self._active_count,
                ]
            )

            # update nodes
            wp.capture_if(self._first, first_capture.graph)

            wp.launch(
                kernel=k_calculate_forces,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, self.particles]
            )

            wp.launch(
                kernel=k_update_grid,
                dim=self.grid.dimensions,
                inputs=[self.grid, self._active_nodes, self._active_count, self.dt]
            )

            wp.launch(
                kernel=k_calculate_grid_collisions,
                dim=self.grid.dimensions,
                inputs=[self.grid, self._active_nodes, self._active_count, self.obstacle_ids, self.dt]
            )

            wp.launch(
                kernel=k_update_deformations,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights, self.dt]
            )

            # g2p
            wp.launch(
                kernel=k_g2p,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights, self.dt]
            )

            # update particles
            wp.launch(
                kernel=k_calculate_particle_collisions,
                dim=len(self.particles),
                inputs=[self.particles, self.obstacle_ids, self.dt]
            )
            wp.launch(
                kernel=k_advect_particle_positions,
                dim=len(self.particles),
                inputs=[self.particles, self.dt]
            )

            # clear
            self.grid.clear()

        return capture.graph

    def update(self):
        wp.capture_launch(self._graph)
        self.t += self.dt


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
def k_normalize_grid(
    grid: Grid,
    active_flags: wp.array[int]
):
    i, j, k = wp.tid()

    mass = grid.masses[i, j, k]
    flat_idx = (i * grid.dimensions[1] + j) * grid.dimensions[2] + k

    if mass > EPSILON:
        grid.velocities[i, j, k] = grid.velocities[i, j, k] / mass
        active_flags[flat_idx] = 1
    else:
        grid.velocities[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        active_flags[flat_idx] = 0


@wp.kernel
def k_collect_active_nodes(
        grid: Grid,
        active_flags: wp.array[int],
        active_offsets: wp.array[int],
        active_nodes: wp.array[wp.vec3i],
        active_count: wp.array[int],
):
    flat_idx = wp.tid()
    dim_x = grid.dimensions[0]
    dim_y = grid.dimensions[1]
    dim_z = grid.dimensions[2]
    total_elements = dim_x * dim_y * dim_z

    if active_flags[flat_idx] == 1:
        offset = active_offsets[flat_idx]

        i = flat_idx // (dim_y * dim_z)
        rem = flat_idx % (dim_y * dim_z)
        j = rem // dim_z
        k = rem % dim_z

        active_nodes[offset] = wp.vec3i(i, j, k)

    if flat_idx == total_elements - 1:
        active_count[0] = active_offsets[flat_idx] + active_flags[flat_idx]


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

    pk_stress = 2.0 * mu * (F_Ep - R_Ep) + lmbd * (J_Ep - 1.0) * cofactor(F_Ep)
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

                wp.atomic_add(grid.forces, i, j, k, force @ grad_weight)


@wp.kernel
def k_update_grid(
    grid: Grid,
    active_nodes: wp.array[wp.vec3i],
    active_count: wp.array[int],
    dt: float
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    # Look up original 3D indices
    i, j, k = active_nodes[active_id]

    mass = grid.masses[i, j, k]
    vel = grid.velocities[i, j, k]

    # Elastic force
    vel += dt * grid.forces[i, j, k] / mass

    # Gravity
    vel += dt * GRAVITY

    grid.new_velocities[i, j, k] = vel


@wp.kernel
def k_calculate_grid_collisions(
    grid: Grid,
    active_nodes: wp.array[wp.vec3i],
    active_count: wp.array[int],
    obstacles: wp.array[wp.uint64],
    dt: float
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    i, j, k = active_nodes[active_id]

    position = grid_index_to_coord(grid, i, j, k)
    velocity = grid.new_velocities[i, j, k]

    for b in range(obstacles.shape[0]):
        test_position = position + dt * velocity
        obs_id = obstacles[b]

        query = wp.mesh_query_point_sign_normal(obs_id, test_position, MAX_COLLISION_DIST, EPSILON)

        if query.result:
            p = wp.mesh_eval_position(obs_id, query.face, query.u, query.v)
            delta = test_position - p
            dist = wp.length(delta) * query.sign

            if dist <= 0.0:
                delta_len = wp.length(delta)

                if delta_len > 1e-6:
                    normal = (delta / delta_len) * query.sign
                else:
                    # Fallback if exactly on the face boundary: calculate normal from vertices
                    v0 = wp.mesh_eval_position(obs_id, query.face, 0.0, 0.0)
                    v1 = wp.mesh_eval_position(obs_id, query.face, 1.0, 0.0)
                    v2 = wp.mesh_eval_position(obs_id, query.face, 0.0, 1.0)
                    normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))

                velocity_normal = wp.dot(velocity, normal)
                velocity_tangent = velocity - velocity_normal * normal

                if velocity_normal > 0.0:
                    continue

                if wp.length(velocity_tangent) <= -COULOMB_FRICTION * velocity_normal:
                    velocity = wp.vec3(0.0, 0.0, 0.0)
                else:
                    velocity = velocity_tangent + COULOMB_FRICTION * velocity_normal * (
                                velocity_tangent / wp.length(velocity_tangent))

    grid.new_velocities[i, j, k] = velocity


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
    F_Ep_tentative = (I + dt * velocity_gradient) @ F_Ep
    F_Pp_tentative = F_Pp
    F_p = F_Ep_tentative @ F_Pp_tentative

    U_p, sigma_p, V_p = safe_svd3(F_Ep_tentative)

    sigma_p_clamped = wp.vec3(0.0)
    low = 1.0 - particles.critical_compressions[p]
    high = 1.0 + particles.critical_stretches[p]
    for i in range(3):
        sigma_p_clamped[i] = wp.clamp(sigma_p[i], low, high)

    sigma_p_clamped_diag = wp.diag(sigma_p_clamped)

    inv_sigma_p_clamped = wp.vec3(
        1.0 / sigma_p_clamped[0],
        1.0 / sigma_p_clamped[1],
        1.0 / sigma_p_clamped[2]
    )
    inv_sigma_p_clamped_diag = wp.diag(inv_sigma_p_clamped)

    particles.elastic_deformations[p] = U_p @ sigma_p_clamped_diag @ wp.transpose(V_p)
    particles.plastic_deformations[p] = V_p @ inv_sigma_p_clamped_diag @ wp.transpose(U_p) @ F_p


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
def k_calculate_particle_collisions(
        particles: Particles,
        boundaries: wp.array[wp.uint64],
        dt: float
):
    p = wp.tid()

    position = particles.positions[p]
    velocity = particles.velocities[p]

    for b in range(boundaries.shape[0]):
        test_position = position + dt * velocity
        boundary_id = boundaries[b]

        query = wp.mesh_query_point_sign_normal(boundary_id, test_position, MAX_COLLISION_DIST, EPSILON)

        if query.result:
            closest_p = wp.mesh_eval_position(boundary_id, query.face, query.u, query.v)
            delta = test_position - closest_p
            dist = wp.length(delta) * query.sign

            if dist <= 0.0:
                delta_len = wp.length(delta)
                normal = wp.vec3(0.0, 0.0, 0.0)

                if delta_len > 1e-6:
                    normal = (delta / delta_len) * query.sign
                else:
                    # Fallback for boundary touch
                    v0 = wp.mesh_eval_position(boundary_id, query.face, 0.0, 0.0)
                    v1 = wp.mesh_eval_position(boundary_id, query.face, 1.0, 0.0)
                    v2 = wp.mesh_eval_position(boundary_id, query.face, 0.0, 1.0)
                    normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))

                velocity_normal = wp.dot(velocity, normal)
                velocity_tangent = velocity - velocity_normal * normal

                if velocity_normal > 0.0:
                    continue

                if wp.length(velocity_tangent) <= -COULOMB_FRICTION * velocity_normal:
                    velocity = wp.vec3(0.0, 0.0, 0.0)
                else:
                    velocity = velocity_tangent + COULOMB_FRICTION * velocity_normal * (
                                velocity_tangent / wp.length(velocity_tangent))

    particles.velocities[p] = velocity


@wp.kernel
def k_advect_particle_positions(particles: Particles, dt: float):
    p = wp.tid()
    particles.positions[p] += dt * particles.velocities[p]
