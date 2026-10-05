from enum import StrEnum
from functools import cached_property

import warp as wp
from warp.optim.linear import CR, LinearOperator, cr

from mlsmpm.constants import (
    COULOMB_FRICTION,
    DEFAULT_CFL,
    DEFAULT_MAX_DT,
    DEFAULT_MIN_DT,
    EPSILON,
    EPSILON_SQ,
    GRAVITY,
    IMPLICIT_BETA,
    MAX_COLLISION_DIST,
    PICFLIP_ALPHA,
    WALL_LAYERS
)
from mlsmpm.grid import Grid, grid_index_to_coord, grid_index_from_flat, grid_index_to_flat
from mlsmpm.particles import Particles
from mlsmpm.utils import (
    bspline_w,
    cofactor,
    cofactor_differential,
    extract_rotation,
    polar_decompose,
    rotation_differential,
    safe_svd3,
)


class Method(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class Solver:
    grid: Grid
    particles: Particles
    obstacles: list[wp.Mesh]

    cfl: float
    min_dt: float
    max_dt: float
    cr_tol: float
    cr_maxiter: int
    t: float = 0

    _dt: wp.array[float]
    _beta_dt2: wp.array[float]
    _max_speed_sq: wp.array[float]
    _min_cell_size: float

    _active_count: wp.array[int]
    _active_flags: wp.array[int]
    _active_offsets: wp.array[int]
    _active_nodes: wp.array[wp.vec3i]

    _hx: wp.array[wp.vec3]
    _rhs: wp.array[wp.vec3]
    _solution: wp.array[wp.vec3]
    _node_sqrt_mass: wp.array[float]
    _v_candidate: wp.array[wp.vec3]
    _operator: LinearOperator
    _cr: CR
    _constrained: wp.array[int]

    _implicit: wp.array[int]

    _graph: wp.Graph

    def _matvec(self):
        def matvec(y, y_other, z, alpha, beta):
            wp.launch(
                kernel=k_solver_unscale_by_sqrt_mass,
                dim=len(y),
                inputs=[
                    self._node_sqrt_mass,
                    self._active_count,
                    y,
                    self._v_candidate,
                ],
            )

            self._hx.zero_()
            wp.launch(
                kernel=k_solver_hessian_action,
                dim=len(self.particles),
                inputs=[
                    self.grid,
                    self.particles,
                    self._active_flags,
                    self._active_offsets,
                    self._constrained,
                    self._v_candidate,
                    self._hx,
                ],
            )

            wp.launch(
                kernel=k_solver_matvec,
                dim=len(y),
                inputs=[
                    self._node_sqrt_mass,
                    self._hx,
                    self._active_count,
                    self._constrained,
                    self._beta_dt2,
                    y,
                    y_other,
                    z,
                    alpha,
                    beta,
                ],
            )

        return matvec

    def __init__(
            self,
            grid: Grid,
            particles: Particles,
            obstacles: list[wp.Mesh],
            method: Method = Method.EXPLICIT,
            cfl: float = DEFAULT_CFL,
            min_dt: float = DEFAULT_MIN_DT,
            max_dt: float = DEFAULT_MAX_DT,
            cr_tol: float = 1e-4,
            cr_maxiter: int = 3000,
    ):
        self.grid = grid
        self.particles = particles
        self.obstacles = obstacles
        self.cfl = cfl
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.cr_tol = cr_tol
        self.cr_maxiter = cr_maxiter

        self._dt = wp.zeros(1, dtype=float)
        self._dt.fill_(self.max_dt)
        self._beta_dt2 = wp.zeros(1, dtype=float)
        self._max_speed_sq = wp.zeros(1, dtype=float)
        self._min_cell_size = self.grid.min_size()

        flat_size = self.grid.flat_dimensions
        self._active_flags = wp.zeros(shape=[flat_size], dtype=wp.int32)
        self._active_offsets = wp.zeros(shape=[flat_size], dtype=wp.int32)
        self._active_nodes = wp.zeros(shape=[flat_size], dtype=wp.vec3i)
        self._active_count = wp.zeros(shape=1, dtype=wp.int32)

        self._first = wp.zeros(1, dtype=int)
        self._first.fill_(1)
        self._implicit = wp.zeros(1, dtype=int)
        self.method = method

        self._rhs = wp.zeros(shape=[flat_size], dtype=wp.vec3)
        self._solution = wp.zeros(shape=[flat_size], dtype=wp.vec3)
        self._node_sqrt_mass = wp.zeros(shape=[flat_size], dtype=float)
        self._v_candidate = wp.zeros(shape=[flat_size], dtype=wp.vec3)
        self._constrained = wp.zeros(shape=[flat_size], dtype=wp.int32)
        self._hx = wp.zeros(shape=[flat_size], dtype=wp.vec3)

        self._operator = LinearOperator(
            shape=(flat_size, flat_size),
            dtype=wp.vec3,
            device="cuda",
            matvec=self._matvec(),
        )

        self._cr = cr(
            self._operator,
            self._rhs,
            self._solution,
            tol=self.cr_tol,
            maxiter=self.cr_maxiter,
            check_every=0,
            run=False,
        )

        _ = self.obstacle_ids

        self._init_particle_volumes()
        self._graph = self._capture_graph()

    @cached_property
    def obstacle_ids(self) -> wp.array[wp.uint64]:
        return wp.array([obs.id for obs in self.obstacles], dtype=wp.uint64)

    @property
    def method(self) -> Method:
        return Method.IMPLICIT if self._implicit.numpy()[0] else Method.EXPLICIT

    @method.setter
    def method(self, method: Method):
        self._implicit.fill_(int(Method(method) == Method.IMPLICIT))

    def _init_particle_volumes(self):
        wp.launch(
            kernel=k_solver_p2g_mass_only,
            dim=len(self.particles),
            inputs=[self.particles, self.grid],
        )
        wp.launch(
            kernel=k_solver_calculate_density,
            dim=len(self.particles),
            inputs=[self.grid, self.particles],
        )
        wp.launch(
            kernel=k_solver_set_initial_volumes,
            dim=len(self.particles),
            inputs=[self.particles],
        )
        self.grid.clear()

    def _implicit_solve(self):
        wp.launch(
            kernel=k_solver_gather_sqrt_mass,
            dim=self.grid.flat_dimensions,
            inputs=[
                self.grid,
                self._active_nodes,
                self._active_count,
                self._node_sqrt_mass,
            ],
        )

        wp.launch(
            kernel=k_solver_gather_rhs,
            dim=self.grid.flat_dimensions,
            inputs=[
                self.grid,
                self._active_nodes,
                self._active_count,
                self._node_sqrt_mass,
                self._rhs,
            ],
        )

        self._solution.assign(self._rhs)

        self._cr()

        wp.launch(
            kernel=k_solver_scatter_solution,
            dim=self.grid.flat_dimensions,
            inputs=[
                self.grid,
                self._active_nodes,
                self._active_count,
                self._node_sqrt_mass,
                self._solution,
            ],
        )

    def _capture_graph(self):
        with wp.ScopedCapture(capture_mode=wp.CaptureMode.RELAXED) as capture:
            # compute dt
            wp.launch(
                kernel=k_solver_compute_max_speed_sq,
                dim=len(self.particles),
                inputs=[self.particles, self._max_speed_sq],
            )

            wp.launch(
                kernel=k_solver_compute_dt,
                dim=1,
                inputs=[
                    self._max_speed_sq,
                    self._min_cell_size,
                    self.cfl,
                    self.max_dt,
                    self.min_dt,
                    self._dt,
                    self._beta_dt2,
                ],
            )

            # p2g
            wp.launch(
                kernel=k_solver_p2g,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._dt],
            )

            # grid normalization
            wp.launch(
                kernel=k_solver_normalize_grid,
                dim=self.grid.dimensions,
                inputs=[self.grid, self._active_flags],
            )
            wp.utils.array_scan(
                self._active_flags, self._active_offsets, inclusive=False
            )
            wp.launch(
                kernel=k_solver_collect_active_nodes,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self._active_flags,
                    self._active_offsets,
                    self._active_nodes,
                    self._active_count,
                ],
            )

            # grid update and collisions
            wp.launch(
                kernel=k_solver_update_grid,
                dim=self.grid.flat_dimensions,
                inputs=[self.grid, self._active_nodes, self._active_count, self._dt],
            )
            wp.launch(
                kernel=k_solver_calculate_grid_collisions,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self._active_nodes,
                    self._active_count,
                    self.obstacle_ids,
                    self._dt,
                    self._constrained,
                ],
            )

            # implicit velocity update
            wp.capture_if(self._implicit, self._implicit_solve)

            # g2p + deformation update + particle collisions + advection
            wp.launch(
                kernel=k_solver_g2p,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self.obstacle_ids, self._dt],
            )

            # clear
            self.grid.clear()
            self._max_speed_sq.zero_()

        return capture.graph

    def update(self):
        wp.capture_launch(self._graph)
        self.t += float(self._dt.numpy()[0])

    def current_dt(self):
        return self._dt


@wp.func
def mu_(
        plastic_deformation: wp.mat33, hardening_coef: float, initial_mu: float
) -> float:
    Jp = wp.determinant(plastic_deformation)
    hardening_mult = wp.exp(hardening_coef * (1.0 - Jp))
    return initial_mu * hardening_mult


@wp.func
def lambda_(
        plastic_deformation: wp.mat33, hardening_coef: float, initial_lambda: float
) -> float:
    Jp = wp.determinant(plastic_deformation)
    hardening_mult = wp.exp(hardening_coef * (1.0 - Jp))
    return initial_lambda * hardening_mult


@wp.func
def compute_weights(
        position: wp.vec3,
        min_coord: wp.vec3,
        cell_size: wp.vec3,
):
    inv_dx = 1.0 / cell_size[0]
    inv_dy = 1.0 / cell_size[1]
    inv_dz = 1.0 / cell_size[2]

    # Normalized particle position relative to the grid origin
    rel_x = (position[0] - min_coord[0]) * inv_dx
    rel_y = (position[1] - min_coord[1]) * inv_dy
    rel_z = (position[2] - min_coord[2]) * inv_dz

    # Base grid node coordinates
    base_i = wp.int32(wp.floor(rel_x)) - 1
    base_j = wp.int32(wp.floor(rel_y)) - 1
    base_k = wp.int32(wp.floor(rel_z)) - 1
    base = wp.vec3i(base_i, base_j, base_k)

    # Offset to base node in normalized grid units
    fx = rel_x - float(base_i)
    fy = rel_y - float(base_j)
    fz = rel_z - float(base_k)

    wx = wp.vec4(0.0)
    wy = wp.vec4(0.0)
    wz = wp.vec4(0.0)

    # Evaluate 1D B-spline weights
    for d in range(4):
        wx[d] = bspline_w(fx - float(d))
        wy[d] = bspline_w(fy - float(d))
        wz[d] = bspline_w(fz - float(d))

    return base, wx, wy, wz


@wp.kernel
def k_solver_p2g_mass_only(particles: Particles, grid: Grid):
    p = wp.tid()
    mass = particles.masses[p]
    base, wx, wy, wz = compute_weights(
        particles.positions[p], grid.min_coord, grid.cell_size
    )
    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk
                if (0 <= i < grid.masses.shape[0] and
                        0 <= j < grid.masses.shape[1] and
                        0 <= k < grid.masses.shape[2]):
                    weight = wx[di] * wy[dj] * wz[dk]
                    wp.atomic_add(grid.masses, i, j, k, weight * mass)


@wp.kernel
def k_solver_calculate_density(
        grid: Grid, particles: Particles
):
    p = wp.tid()

    base, wx, wy, wz = compute_weights(
        particles.positions[p], grid.min_coord, grid.cell_size
    )

    density = float(0.0)
    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                        i < 0
                        or i >= grid.masses.shape[0]
                        or j < 0
                        or j >= grid.masses.shape[1]
                        or k < 0
                        or k >= grid.masses.shape[2]
                ):
                    continue

                weight = wx[di] * wy[dj] * wz[dk]
                density += (
                        weight
                        * grid.masses[i, j, k]
                        / (grid.cell_size[0] * grid.cell_size[1] * grid.cell_size[2])
                )

    particles.densities[p] = density


@wp.kernel
def k_solver_set_initial_volumes(particles: Particles):
    p = wp.tid()
    particles.volumes[p] = particles.masses[p] / particles.densities[p]


@wp.kernel
def k_solver_compute_max_speed_sq(
        particles: Particles,
        max_speed_sq: wp.array[float]
):
    p = wp.tid()
    v = particles.velocities[p]
    wp.atomic_max(max_speed_sq, 0, wp.dot(v, v))


@wp.kernel
def k_solver_compute_dt(
        max_speed_sq: wp.array[float],
        cell_size_min: float,
        cfl: float,
        max_timestep: float,
        min_timestep: float,
        dt: wp.array[float],
        beta_dt2: wp.array[float]
):
    speed = wp.sqrt(wp.max(max_speed_sq[0], EPSILON_SQ))
    dt[0] = wp.max(min_timestep, wp.min(cfl * cell_size_min / speed, max_timestep))
    beta_dt2[0] = IMPLICIT_BETA * dt[0] * dt[0]


@wp.kernel
def k_solver_p2g(particles: Particles, grid: Grid, dt_: wp.array[float]):
    p = wp.tid()
    dt = dt_[0]

    # Get particle properties
    mass = particles.masses[p]
    velocity = particles.velocities[p]
    position = particles.positions[p]
    B = particles.affine_matrices[p]
    FE = particles.elastic_deformations[p]
    FP = particles.plastic_deformations[p]
    hardening_coef = particles.hardening_coefs[p]
    mu = mu_(FP, hardening_coef, particles.mus[p])
    lmbd = lambda_(FP, hardening_coef, particles.lambdas[p])
    volume_0 = particles.volumes[p]

    # Compute PK1 stress and force
    RE = extract_rotation(FE)
    JE = wp.determinant(FE)
    pk_stress = 2.0 * mu * (FE - RE) + lmbd * (JE - 1.0) * cofactor(FE)
    force = -1.0 * (volume_0 * pk_stress @ wp.transpose(FE))

    # D_inverse for cubic B-spline
    D_inv = wp.vec3(
        3.0 / (grid.cell_size[0] * grid.cell_size[0]),
        3.0 / (grid.cell_size[1] * grid.cell_size[1]),
        3.0 / (grid.cell_size[2] * grid.cell_size[2])
    )
    D_inv_diag = wp.diag(D_inv)

    # Momentum contribution
    Q = (mass * B + dt * force) @ D_inv_diag

    # Get the weights for the particle
    base, wx, wy, wz = compute_weights(
        particles.positions[p], grid.min_coord, grid.cell_size
    )

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (i < 0 or i >= grid.masses.shape[0] or
                        j < 0 or j >= grid.masses.shape[1] or
                        k < 0 or k >= grid.masses.shape[2]):
                    continue

                weight = wx[di] * wy[dj] * wz[dk]
                offset = grid_index_to_coord(grid, i, j, k) - position

                wp.atomic_add(grid.masses, i, j, k, weight * mass)
                wp.atomic_add(
                    grid.velocities, i, j, k,
                    weight * (mass * velocity + Q @ offset)
                )


@wp.kernel
def k_solver_normalize_grid(grid: Grid, active_flags: wp.array[int]):
    i, j, k = wp.tid()

    mass = grid.masses[i, j, k]
    idx = grid_index_to_flat(grid, i, j, k)

    if mass > EPSILON:
        grid.velocities[i, j, k] = grid.velocities[i, j, k] / mass
        active_flags[idx] = 1
    else:
        grid.velocities[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        active_flags[idx] = 0


@wp.kernel
def k_solver_collect_active_nodes(
        grid: Grid,
        active_flags: wp.array[int],
        active_offsets: wp.array[int],
        active_nodes: wp.array[wp.vec3i],
        active_count: wp.array[int],
):
    flat_idx = wp.tid()
    dim_x = int(grid.dimensions[0])
    dim_y = int(grid.dimensions[1])
    dim_z = int(grid.dimensions[2])
    total_elements = dim_x * dim_y * dim_z

    if active_flags[flat_idx] == 1:
        offset = active_offsets[flat_idx]
        (i, j, k) = grid_index_from_flat(grid, flat_idx)

        active_nodes[offset] = wp.vec3i(i, j, k)

    if flat_idx == total_elements - 1:
        active_count[0] = active_offsets[flat_idx] + active_flags[flat_idx]


@wp.kernel
def k_solver_update_grid(
        grid: Grid,
        active_nodes: wp.array[wp.vec3i],
        active_count: wp.array[int],
        _dt: wp.array[float],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    dt = _dt[0]
    coord = active_nodes[active_id]
    i, j, k = coord[0], coord[1], coord[2]

    vel = grid.velocities[i, j, k]
    vel += dt * GRAVITY

    grid.new_velocities[i, j, k] = vel


@wp.kernel
def k_solver_calculate_grid_collisions(
        grid: Grid,
        active_nodes: wp.array[wp.vec3i],
        active_count: wp.array[int],
        obstacles: wp.array[wp.uint64],
        _dt: wp.array[float],
        constrained: wp.array[int],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return
    dt = _dt[0]

    coord = active_nodes[active_id]
    i, j, k = coord[0], coord[1], coord[2]

    position = grid_index_to_coord(grid, i, j, k)
    velocity = grid.new_velocities[i, j, k]
    is_constrained = wp.int32(0)

    for b in range(obstacles.shape[0]):
        test_position = position + dt * velocity
        obs_id = obstacles[b]

        query = wp.mesh_query_point_sign_normal(
            obs_id, test_position, MAX_COLLISION_DIST, EPSILON
        )

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

                if velocity_normal >= 0.0:
                    continue

                is_constrained = wp.int32(1)

                if (
                        wp.length(velocity_tangent)
                        <= -COULOMB_FRICTION * velocity_normal
                ):
                    velocity = wp.vec3(0.0, 0.0, 0.0)
                else:
                    velocity = (
                            velocity_tangent
                            + COULOMB_FRICTION
                            * velocity_normal
                            * (velocity_tangent / wp.length(velocity_tangent))
                    )

    # domain walls: separating slip condition on the outer WALL_LAYERS node layers
    idx = wp.vec3i(i, j, k)
    for a in range(3):
        if idx[a] < WALL_LAYERS and velocity[a] < 0.0:
            velocity[a] = 0.0
            is_constrained = wp.int32(1)
        if idx[a] >= int(grid.dimensions[a]) - WALL_LAYERS and velocity[a] > 0.0:
            velocity[a] = 0.0
            is_constrained = wp.int32(1)

    grid.new_velocities[i, j, k] = velocity
    constrained[active_id] = is_constrained


@wp.kernel
def k_solver_hessian_action(
        grid: Grid,
        particles: Particles,
        active_flags: wp.array[int],
        active_offsets: wp.array[int],
        constrained: wp.array[int],
        x: wp.array[wp.vec3],
        hx: wp.array[wp.vec3],
):
    p = wp.tid()

    position = particles.positions[p]

    base, wx, wy, wz = compute_weights(
        position, grid.min_coord, grid.cell_size
    )

    F_Ep = particles.elastic_deformations[p]
    F_Pp = particles.plastic_deformations[p]
    xi = particles.hardening_coefs[p]
    mu = mu_(F_Pp, xi, particles.mus[p])
    lmbd = lambda_(F_Pp, xi, particles.lambdas[p])
    V_0 = particles.volumes[p]

    dim_y = int(grid.dimensions[1])
    dim_z = int(grid.dimensions[2])

    A = wp.mat33(0.0)
    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                        i < 0
                        or i >= grid.masses.shape[0]
                        or j < 0
                        or j >= grid.masses.shape[1]
                        or k < 0
                        or k >= grid.masses.shape[2]
                ):
                    continue

                flat_idx = (i * dim_y + j) * dim_z + k
                if active_flags[flat_idx] == 0:
                    continue
                active_id = active_offsets[flat_idx]
                if constrained[active_id] == 1:
                    continue

                D_inverse = wp.vec3(
                    3.0 / (grid.cell_size[0] * grid.cell_size[0]),
                    3.0 / (grid.cell_size[1] * grid.cell_size[1]),
                    3.0 / (grid.cell_size[2] * grid.cell_size[2])
                )

                weight = wx[di] * wy[dj] * wz[dk]
                offset = grid_index_to_coord(grid, i, j, k) - position
                grad_weight = weight * wp.cw_mul(D_inverse, offset)

                A += wp.outer(x[active_id], grad_weight)

    dF = A @ F_Ep

    R, S = polar_decompose(F_Ep)
    dR = rotation_differential(dF, R, S)

    cof_F = cofactor(F_Ep)
    dcof_F = cofactor_differential(F_Ep, dF)

    contraction = 0.0
    for r in range(3):
        for c in range(3):
            contraction += cof_F[r, c] * dF[r, c]

    J = wp.determinant(F_Ep)

    dP = (
            2.0 * mu * (dF - dR)
            + lmbd * contraction * cof_F
            + lmbd * (J - 1.0) * dcof_F
    )
    force = V_0 * dP @ wp.transpose(F_Ep)

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (
                        i < 0
                        or i >= grid.masses.shape[0]
                        or j < 0
                        or j >= grid.masses.shape[1]
                        or k < 0
                        or k >= grid.masses.shape[2]
                ):
                    continue

                flat_idx = (i * dim_y + j) * dim_z + k
                if active_flags[flat_idx] == 0:
                    continue
                active_id = active_offsets[flat_idx]
                if constrained[active_id] == 1:
                    continue

                D_inverse = wp.vec3(
                    3.0 / (grid.cell_size[0] * grid.cell_size[0]),
                    3.0 / (grid.cell_size[1] * grid.cell_size[1]),
                    3.0 / (grid.cell_size[2] * grid.cell_size[2])
                )

                weight = wx[di] * wy[dj] * wz[dk]
                offset = grid_index_to_coord(grid, i, j, k) - position
                grad_weight = weight * wp.cw_mul(D_inverse, offset)

                wp.atomic_add(hx, active_id, force @ grad_weight)


@wp.kernel
def k_solver_unscale_by_sqrt_mass(
        sqrt_mass: wp.array[float],
        active_count: wp.array[int],
        y: wp.array[wp.vec3],
        x_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    if tid >= active_count[0]:
        x_out[tid] = wp.vec3(0.0)
        return
    x_out[tid] = y[tid] / sqrt_mass[tid]


@wp.kernel
def k_solver_matvec(
        sqrt_mass: wp.array[float],
        hx: wp.array[wp.vec3],
        active_count: wp.array[int],
        constrained: wp.array[int],
        _beta_dt2: wp.array[float],
        y: wp.array[wp.vec3],
        y_other: wp.array[wp.vec3],
        z: wp.array[wp.vec3],
        alpha: float,
        beta: float,
):
    tid = wp.tid()
    beta_dt2 = _beta_dt2[0]
    if tid >= active_count[0] or constrained[tid] == 1:
        z[tid] = alpha * y[tid] + beta * y_other[tid]
        return
    z[tid] = (
            alpha * (y[tid] + beta_dt2 * hx[tid] / sqrt_mass[tid])
            + beta * y_other[tid]
    )


@wp.kernel
def k_solver_gather_sqrt_mass(
        grid: Grid,
        active_nodes: wp.array[wp.vec3i],
        active_count: wp.array[int],
        sqrt_mass: wp.array[float],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j, k = coord[0], coord[1], coord[2]

    sqrt_mass[active_id] = wp.sqrt(grid.masses[i, j, k])


@wp.kernel
def k_solver_gather_rhs(
        grid: Grid,
        active_nodes: wp.array[wp.vec3i],
        active_count: wp.array[int],
        sqrt_mass: wp.array[float],
        rhs: wp.array[wp.vec3],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j, k = coord[0], coord[1], coord[2]

    v_star = grid.new_velocities[i, j, k]
    rhs[active_id] = sqrt_mass[active_id] * v_star


@wp.kernel
def k_solver_scatter_solution(
        grid: Grid,
        active_nodes: wp.array[wp.vec3i],
        active_count: wp.array[int],
        sqrt_mass: wp.array[float],
        solution: wp.array[wp.vec3],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j, k = coord[0], coord[1], coord[2]

    grid.new_velocities[i, j, k] = solution[active_id] / sqrt_mass[active_id]


@wp.kernel
def k_solver_g2p(particles: Particles,
                 grid: Grid,
                 boundaries: wp.array[wp.uint64],
                 _dt: wp.array[float],
                 ):
    p = wp.tid()
    dt = _dt[0]
    position = particles.positions[p]

    base, wx, wy, wz = compute_weights(
        position, grid.min_coord, grid.cell_size
    )

    velocity = wp.vec3(0.0)
    B = wp.mat33(0.0)

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base[0] + di
                j = base[1] + dj
                k = base[2] + dk

                if (i < 0 or i >= grid.masses.shape[0] or
                        j < 0 or j >= grid.masses.shape[1] or
                        k < 0 or k >= grid.masses.shape[2]):
                    continue

                weight = wx[di] * wy[dj] * wz[dk]
                node_vel = grid.new_velocities[i, j, k]
                offset = grid_index_to_coord(grid, i, j, k) - position

                velocity += weight * node_vel
                B += weight * wp.outer(node_vel, offset)

    # Update particle velocity and affine matrix
    particles.velocities[p] = velocity
    particles.affine_matrices[p] = B

    # Update deformation gradients
    D_inv = wp.vec3(
        3.0 / (grid.cell_size[0] * grid.cell_size[0]),
        3.0 / (grid.cell_size[1] * grid.cell_size[1]),
        3.0 / (grid.cell_size[2] * grid.cell_size[2]),
    )
    C = B @ wp.diag(D_inv)
    I = wp.identity(3, dtype=wp.float32)

    F_Ep = particles.elastic_deformations[p]
    F_Pp = particles.plastic_deformations[p]

    F_E_tentative = (I + dt * C) @ F_Ep
    F = F_E_tentative @ F_Pp

    # SVD and clamping for plasticity
    U, sigma, V = safe_svd3(F_E_tentative)
    low = 1.0 - particles.critical_compressions[p]
    high = 1.0 + particles.critical_stretches[p]
    sigma_clamped = wp.vec3(
        wp.clamp(sigma[0], low, high),
        wp.clamp(sigma[1], low, high),
        wp.clamp(sigma[2], low, high),
    )
    inv_sigma_clamped = wp.vec3(
        1.0 / sigma_clamped[0],
        1.0 / sigma_clamped[1],
        1.0 / sigma_clamped[2],
    )

    particles.elastic_deformations[p] = U @ wp.diag(sigma_clamped) @ wp.transpose(V)
    particles.plastic_deformations[p] = V @ wp.diag(inv_sigma_clamped) @ wp.transpose(U) @ F

    # Particle collision handling
    for b in range(boundaries.shape[0]):
        test_pos = position + dt * velocity
        boundary_id = boundaries[b]
        query = wp.mesh_query_point_sign_normal(
            boundary_id, test_pos, MAX_COLLISION_DIST, EPSILON
        )
        if query.result:
            closest_p = wp.mesh_eval_position(boundary_id, query.face, query.u, query.v)
            delta = test_pos - closest_p
            dist = wp.length(delta) * query.sign
            if dist <= 0.0:
                delta_len = wp.length(delta)
                normal = wp.vec3(0.0)
                if delta_len > 1e-6:
                    normal = (delta / delta_len) * query.sign
                else:
                    v0 = wp.mesh_eval_position(boundary_id, query.face, 0.0, 0.0)
                    v1 = wp.mesh_eval_position(boundary_id, query.face, 1.0, 0.0)
                    v2 = wp.mesh_eval_position(boundary_id, query.face, 0.0, 1.0)
                    normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))

                v_n = wp.dot(velocity, normal)
                if v_n < 0.0:
                    v_t = velocity - v_n * normal
                    v_t_len = wp.length(v_t)
                    if v_t_len <= -COULOMB_FRICTION * v_n:
                        velocity = wp.vec3(0.0)
                    else:
                        velocity = v_t + COULOMB_FRICTION * v_n * (v_t / v_t_len)

    particles.velocities[p] = velocity

    # Advect particle positions
    particles.positions[p] = position + dt * velocity
