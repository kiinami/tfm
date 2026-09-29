from enum import StrEnum
from functools import cached_property

import warp as wp
from warp.optim.linear import CR, LinearOperator, cr

from mpm2d.constants import (
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
)
from mpm2d.grid import Grid, grid_index_to_coord, grid_index_from_flat, grid_index_to_flat
from mpm2d.particles import Particles
from mpm2d.utils import (
    Obstacles,
    Polygon,
    bspline_dw,
    bspline_w,
    cofactor,
    cofactor_differential,
    extract_rotation,
    polar_decompose,
    polygon_edge_normal,
    polygon_query_point_sign_normal,
    rotation_differential,
    safe_svd2,
)


class Method(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


@wp.struct
class Weights:
    base: wp.array[wp.vec2i]
    wx: wp.array[wp.vec4]
    wy: wp.array[wp.vec4]
    dwx: wp.array[wp.vec4]
    dwy: wp.array[wp.vec4]

    def init(self, n: int):
        self.base = wp.empty(n, dtype=wp.vec2i)
        self.wx = wp.empty(n, dtype=wp.vec4)
        self.wy = wp.empty(n, dtype=wp.vec4)
        self.dwx = wp.empty(n, dtype=wp.vec4)
        self.dwy = wp.empty(n, dtype=wp.vec4)


class Solver:
    grid: Grid
    particles: Particles
    obstacles: list[Polygon]

    cfl: float
    min_dt: float
    max_dt: float
    cr_tol: float
    cr_maxiter: int
    t: float = 0

    _weights: Weights

    _dt: wp.array[float]
    _beta_dt2: wp.array[float]
    _max_speed_sq: wp.array[float]
    _min_cell_size: float

    _active_count: wp.array[int]
    _active_flags: wp.array[int]
    _active_offsets: wp.array[int]
    _active_nodes: wp.array[wp.vec2i]

    _hx: wp.array[wp.vec2]
    _rhs: wp.array[wp.vec2]
    _solution: wp.array[wp.vec2]
    _node_sqrt_mass: wp.array[float]
    _v_candidate: wp.array[wp.vec2]
    _operator: LinearOperator
    _cr: CR
    _constrained: wp.array[int]

    _first: wp.array[int]
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
                    self._weights,
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
            obstacles: list[Polygon],
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

        self._weights = Weights()
        self._weights.init(len(particles))

        self._dt = wp.zeros(1, dtype=float)
        self._dt.fill_(self.max_dt)
        self._beta_dt2 = wp.zeros(1, dtype=float)
        self._max_speed_sq = wp.zeros(1, dtype=float)
        self._min_cell_size = self.grid.min_size()

        flat_size = self.grid.flat_dimensions
        self._active_flags = wp.zeros(shape=[flat_size], dtype=wp.int32)
        self._active_offsets = wp.zeros(shape=[flat_size], dtype=wp.int32)
        self._active_nodes = wp.zeros(shape=[flat_size], dtype=wp.vec2i)
        self._active_count = wp.zeros(shape=1, dtype=wp.int32)

        self._first = wp.zeros(1, dtype=int)
        self._first.fill_(1)
        self._implicit = wp.zeros(1, dtype=int)
        self.method = method

        self._rhs = wp.zeros(shape=[flat_size], dtype=wp.vec2)
        self._solution = wp.zeros(shape=[flat_size], dtype=wp.vec2)
        self._node_sqrt_mass = wp.zeros(shape=[flat_size], dtype=float)
        self._v_candidate = wp.zeros(shape=[flat_size], dtype=wp.vec2)
        self._constrained = wp.zeros(shape=[flat_size], dtype=wp.int32)
        self._hx = wp.zeros(shape=[flat_size], dtype=wp.vec2)

        self._operator = LinearOperator(
            shape=(flat_size, flat_size),
            dtype=wp.vec2,
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

        _ = self.packed_obstacles

        self._graph = self._capture_graph()

    @cached_property
    def packed_obstacles(self) -> Obstacles:
        packed = Obstacles()
        packed.init(self.obstacles)
        return packed

    @property
    def method(self) -> Method:
        return Method.IMPLICIT if self._implicit.numpy()[0] else Method.EXPLICIT

    @method.setter
    def method(self, method: Method):
        self._implicit.fill_(int(Method(method) == Method.IMPLICIT))

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
        with wp.ScopedCapture(capture_mode=wp.CaptureMode.RELAXED) as first_capture:
            wp.launch(
                kernel=k_solver_set_initial_volumes,
                dim=len(self.particles),
                inputs=[self.particles],
            )

            self._first.fill_(0)

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
                kernel=k_solver_compute_weights,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights],
            )

            wp.launch(
                kernel=k_solver_p2g,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights],
            )

            wp.launch(
                kernel=k_solver_calculate_density,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, self.particles],
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

            # update nodes
            wp.capture_if(self._first, first_capture.graph)

            wp.launch(
                kernel=k_solver_calculate_forces,
                dim=len(self.particles),
                inputs=[self.grid, self._weights, self.particles],
            )

            wp.launch(
                kernel=k_solver_update_grid,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self._active_nodes,
                    self._active_count,
                    self._dt,
                ],
            )

            wp.launch(
                kernel=k_solver_calculate_grid_collisions,
                dim=self.grid.flat_dimensions,
                inputs=[
                    self.grid,
                    self._active_nodes,
                    self._active_count,
                    self.packed_obstacles,
                    self._dt,
                    self._constrained,
                ],
            )

            # implicit velocity update, skipped when explicit
            wp.capture_if(self._implicit, self._implicit_solve)

            wp.launch(
                kernel=k_solver_update_deformations,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights, self._dt],
            )

            # g2p
            wp.launch(
                kernel=k_solver_g2p,
                dim=len(self.particles),
                inputs=[self.particles, self.grid, self._weights],
            )

            # update particles
            wp.launch(
                kernel=k_solver_calculate_particle_collisions,
                dim=len(self.particles),
                inputs=[self.particles, self.packed_obstacles, self._dt],
            )
            wp.launch(
                kernel=k_solver_advect_particle_positions,
                dim=len(self.particles),
                inputs=[self.particles, self._dt],
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
        plastic_deformation: wp.mat22, hardening_coef: float, initial_mu: float
) -> float:
    Jp = wp.determinant(plastic_deformation)
    hardening_mult = wp.exp(hardening_coef * (1.0 - Jp))
    return initial_mu * hardening_mult


@wp.func
def lambda_(
        plastic_deformation: wp.mat22, hardening_coef: float, initial_lambda: float
) -> float:
    Jp = wp.determinant(plastic_deformation)
    hardening_mult = wp.exp(hardening_coef * (1.0 - Jp))
    return initial_lambda * hardening_mult


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

            wp.atomic_add(grid.masses, i, j, weight * mass)
            wp.atomic_add(
                grid.velocities, i, j, weight * mass * velocity
            )


@wp.kernel
def k_solver_normalize_grid(grid: Grid, active_flags: wp.array[int]):
    i, j = wp.tid()

    mass = grid.masses[i, j]
    idx = grid_index_to_flat(grid, i, j)

    if mass > EPSILON:
        grid.velocities[i, j] = grid.velocities[i, j] / mass
        active_flags[idx] = 1
    else:
        grid.velocities[i, j] = wp.vec2(0.0, 0.0)
        active_flags[idx] = 0


@wp.kernel
def k_solver_collect_active_nodes(
        grid: Grid,
        active_flags: wp.array[int],
        active_offsets: wp.array[int],
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[int],
):
    flat_idx = wp.tid()
    dim_x = int(grid.dimensions[0])
    dim_y = int(grid.dimensions[1])
    total_elements = dim_x * dim_y

    if active_flags[flat_idx] == 1:
        offset = active_offsets[flat_idx]
        (i, j) = grid_index_from_flat(grid, flat_idx)

        active_nodes[offset] = wp.vec2i(i, j)

    if flat_idx == total_elements - 1:
        active_count[0] = active_offsets[flat_idx] + active_flags[flat_idx]


@wp.kernel
def k_solver_calculate_density(
        grid: Grid, weights: Weights, particles: Particles
):
    p = wp.tid()

    base = weights.base[p]
    wx = weights.wx[p]
    wy = weights.wy[p]

    density = float(0.0)
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
            density += (
                    weight
                    * grid.masses[i, j]
                    / (grid.cell_size[0] * grid.cell_size[1])
            )

    particles.densities[p] = density


@wp.kernel
def k_solver_set_initial_volumes(particles: Particles):
    p = wp.tid()
    particles.volumes[p] = particles.masses[p] / particles.densities[p]


@wp.kernel
def k_solver_calculate_forces(grid: Grid, weights: Weights, particles: Particles):
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
    mu = mu_(F_Pp, xi, mu_0)
    lambda_0 = particles.lambdas[p]
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

            grad_weight = wp.vec2(
                dwx[di] * wy[dj],
                wx[di] * dwy[dj],
            )

            wp.atomic_add(grid.forces, i, j, force @ grad_weight)


@wp.kernel
def k_solver_update_grid(
        grid: Grid,
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[int],
        _dt: wp.array[float],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    dt = _dt[0]

    coord = active_nodes[active_id]
    i, j = coord[0], coord[1]

    mass = grid.masses[i, j]
    vel = grid.velocities[i, j]

    vel += dt * (GRAVITY + grid.forces[i, j] / mass)

    grid.new_velocities[i, j] = vel


@wp.kernel
def k_solver_calculate_grid_collisions(
        grid: Grid,
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[int],
        obstacles: Obstacles,
        _dt: wp.array[float],
        constrained: wp.array[int],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return
    dt = _dt[0]

    coord = active_nodes[active_id]
    i, j = coord[0], coord[1]

    position = grid_index_to_coord(grid, i, j)
    velocity = grid.new_velocities[i, j]
    is_constrained = wp.int32(0)

    for b in range(obstacles.offsets.shape[0] - 1):
        test_position = position + dt * velocity

        query = polygon_query_point_sign_normal(
            obstacles, b, test_position, MAX_COLLISION_DIST
        )

        if query.result:
            p = query.point
            delta = test_position - p
            dist = wp.length(delta) * query.sign

            if dist <= 0.0:
                delta_len = wp.length(delta)

                if delta_len > 1e-6:
                    normal = (delta / delta_len) * query.sign
                else:
                    # Fallback if exactly on the edge: use the edge normal
                    normal = polygon_edge_normal(obstacles, b, query.edge)

                velocity_normal = wp.dot(velocity, normal)
                velocity_tangent = velocity - velocity_normal * normal

                if velocity_normal >= 0.0:
                    continue

                is_constrained = wp.int32(1)

                if (
                        wp.length(velocity_tangent)
                        <= -COULOMB_FRICTION * velocity_normal
                ):
                    velocity = wp.vec2(0.0, 0.0)
                else:
                    velocity = (
                            velocity_tangent
                            + COULOMB_FRICTION
                            * velocity_normal
                            * (velocity_tangent / wp.length(velocity_tangent))
                    )

    grid.new_velocities[i, j] = velocity
    constrained[active_id] = is_constrained


@wp.kernel
def k_solver_hessian_action(
        grid: Grid,
        weights: Weights,
        particles: Particles,
        active_flags: wp.array[int],
        active_offsets: wp.array[int],
        constrained: wp.array[int],
        x: wp.array[wp.vec2],
        hx: wp.array[wp.vec2],
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
    mu = mu_(F_Pp, xi, particles.mus[p])
    lmbd = lambda_(F_Pp, xi, particles.lambdas[p])
    V_0 = particles.volumes[p]

    A = wp.mat22(0.0)
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

            flat_idx = grid_index_to_flat(grid, i, j)
            if active_flags[flat_idx] == 0:
                continue
            active_id = active_offsets[flat_idx]
            if constrained[active_id] == 1:
                continue

            grad_weight = wp.vec2(
                dwx[di] * wy[dj],
                wx[di] * dwy[dj],
            )

            A += wp.outer(x[active_id], grad_weight)

    dF = A @ F_Ep

    R, S = polar_decompose(F_Ep)
    dR = rotation_differential(dF, R, S)

    cof_F = cofactor(F_Ep)
    dcof_F = cofactor_differential(F_Ep, dF)

    contraction = 0.0
    for r in range(2):
        for c in range(2):
            contraction += cof_F[r, c] * dF[r, c]

    J = wp.determinant(F_Ep)

    dP = (
            2.0 * mu * (dF - dR)
            + lmbd * contraction * cof_F
            + lmbd * (J - 1.0) * dcof_F
    )
    force = V_0 * dP @ wp.transpose(F_Ep)

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

            flat_idx = grid_index_to_flat(grid, i, j)
            if active_flags[flat_idx] == 0:
                continue
            active_id = active_offsets[flat_idx]
            if constrained[active_id] == 1:
                continue

            grad_weight = wp.vec2(
                dwx[di] * wy[dj],
                wx[di] * dwy[dj],
            )

            wp.atomic_add(hx, active_id, force @ grad_weight)


@wp.kernel
def k_solver_unscale_by_sqrt_mass(
        sqrt_mass: wp.array[float],
        active_count: wp.array[int],
        y: wp.array[wp.vec2],
        x_out: wp.array[wp.vec2],
):
    tid = wp.tid()
    if tid >= active_count[0]:
        x_out[tid] = wp.vec2(0.0)
        return
    x_out[tid] = y[tid] / sqrt_mass[tid]


@wp.kernel
def k_solver_matvec(
        sqrt_mass: wp.array[float],
        hx: wp.array[wp.vec2],
        active_count: wp.array[int],
        constrained: wp.array[int],
        _beta_dt2: wp.array[float],
        y: wp.array[wp.vec2],
        y_other: wp.array[wp.vec2],
        z: wp.array[wp.vec2],
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
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[int],
        sqrt_mass: wp.array[float],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j = coord[0], coord[1]

    sqrt_mass[active_id] = wp.sqrt(grid.masses[i, j])


@wp.kernel
def k_solver_gather_rhs(
        grid: Grid,
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[int],
        sqrt_mass: wp.array[float],
        rhs: wp.array[wp.vec2],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j = coord[0], coord[1]

    v_star = grid.new_velocities[i, j]
    rhs[active_id] = sqrt_mass[active_id] * v_star


@wp.kernel
def k_solver_scatter_solution(
        grid: Grid,
        active_nodes: wp.array[wp.vec2i],
        active_count: wp.array[int],
        sqrt_mass: wp.array[float],
        solution: wp.array[wp.vec2],
):
    active_id = wp.tid()
    if active_id >= active_count[0]:
        return

    coord = active_nodes[active_id]
    i, j = coord[0], coord[1]

    grid.new_velocities[i, j] = solution[active_id] / sqrt_mass[active_id]


@wp.kernel
def k_solver_update_deformations(
        particles: Particles, grid: Grid, weights: Weights, _dt: wp.array[float]
):
    p = wp.tid()
    dt = _dt[0]

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

            grad_weight = wp.vec2(
                dwx[di] * wy[dj],
                wx[di] * dwy[dj],
            )

            new_velocity = grid.new_velocities[i, j]

            velocity_gradient += wp.outer(new_velocity, grad_weight)

    I = wp.identity(2, dtype=wp.float32)
    F_E_tentative = (I + dt * velocity_gradient) @ F_Ep
    F_P_tentative = F_Pp
    F = F_E_tentative @ F_P_tentative

    U, sigma, V = safe_svd2(F_E_tentative)

    sigma_clamped = wp.vec2(0.0)
    low = 1.0 - particles.critical_compressions[p]
    high = 1.0 + particles.critical_stretches[p]
    for i in range(2):
        sigma_clamped[i] = wp.clamp(sigma[i], low, high)

    sigma_clamped_diag = wp.diag(sigma_clamped)

    inv_sigma_clamped = wp.vec2(
        1.0 / sigma_clamped[0],
        1.0 / sigma_clamped[1],
    )
    inv_sigma_clamped_diag = wp.diag(inv_sigma_clamped)

    particles.elastic_deformations[p] = (
            U @ sigma_clamped_diag @ wp.transpose(V)
    )
    particles.plastic_deformations[p] = (
            V @ inv_sigma_clamped_diag @ wp.transpose(U) @ F
    )


@wp.kernel
def k_solver_g2p(particles: Particles, grid: Grid, weights: Weights):
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

            v_pic += node_new_velocity * weight
            v_flip += (node_new_velocity - node_velocity) * weight

    particles.velocities[p] = (
                                      1.0 - PICFLIP_ALPHA
                              ) * v_pic + PICFLIP_ALPHA * v_flip


@wp.kernel
def k_solver_calculate_particle_collisions(
        particles: Particles, boundaries: Obstacles, _dt: wp.array[float]
):
    p = wp.tid()
    dt = _dt[0]

    position = particles.positions[p]
    velocity = particles.velocities[p]

    for b in range(boundaries.offsets.shape[0] - 1):
        test_position = position + dt * velocity

        query = polygon_query_point_sign_normal(
            boundaries, b, test_position, MAX_COLLISION_DIST
        )

        if query.result:
            closest_p = query.point
            delta = test_position - closest_p
            dist = wp.length(delta) * query.sign

            if dist <= 0.0:
                delta_len = wp.length(delta)
                normal = wp.vec2(0.0, 0.0)

                if delta_len > 1e-6:
                    normal = (delta / delta_len) * query.sign
                else:
                    # Fallback for boundary touch
                    normal = polygon_edge_normal(boundaries, b, query.edge)

                velocity_normal = wp.dot(velocity, normal)
                velocity_tangent = velocity - velocity_normal * normal

                if velocity_normal > 0.0:
                    continue

                if (
                        wp.length(velocity_tangent)
                        <= -COULOMB_FRICTION * velocity_normal
                ):
                    velocity = wp.vec2(0.0, 0.0)
                else:
                    velocity = (
                            velocity_tangent
                            + COULOMB_FRICTION
                            * velocity_normal
                            * (velocity_tangent / wp.length(velocity_tangent))
                    )

    particles.velocities[p] = velocity


@wp.kernel
def k_solver_advect_particle_positions(particles: Particles, _dt: wp.array[float]):
    p = wp.tid()
    dt = _dt[0]
    particles.positions[p] += dt * particles.velocities[p]
