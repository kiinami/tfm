import math

import numpy as np
import warp as wp

from mpm_implicit.constants import (
    DEFAULT_CRITICAL_COMPRESSION,
    DEFAULT_CRITICAL_STRETCH,
    DEFAULT_DENSITY,
    DEFAULT_HARDENING_COEFFICIENT,
    DEFAULT_POISSON_RATIO,
    DEFAULT_YOUNG_MODULUS, VOLUME_EPSILON, DEFAULT_PARTICLE_DIAMETER
)

_FIELDS = {
    "volumes": wp.float32,
    "masses": wp.float32,
    "critical_compressions": wp.float32,
    "critical_stretches": wp.float32,
    "hardening_coefs": wp.float32,
    "mus": wp.float32,
    "lambdas": wp.float32,
    "positions": wp.vec3,
    "velocities": wp.vec3,
    "elastic_deformations": wp.mat33,
    "plastic_deformations": wp.mat33,
}


@wp.struct
class Particle:
    volume: float
    mass: float
    position: wp.vec3
    velocity: wp.vec3
    elastic_deformation: wp.mat33
    plastic_deformation: wp.mat33

    critical_compression: float = DEFAULT_CRITICAL_COMPRESSION
    critical_stretch: float = DEFAULT_CRITICAL_STRETCH
    hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT
    initial_young_modulus: float = DEFAULT_YOUNG_MODULUS
    poisson_ratio: float = DEFAULT_POISSON_RATIO


@wp.struct
class Particles:
    """
    Structure containing the data for all particles

    Attributes:
        volumes (float, constant): The initial volume of the particles, assigned on the first iteration
        masses (float, constant): The mass of the particles
        critical_compressions (float, constant): The compression threshold
        critical_stretches (float, constant): The stretch threshold
        hardening_coefs (float, constant): A coefficient that defines how fast the material breaks once yielding
        initial_young_moduli (float, constant): The overall stiffness of the material
        poisson_ratios (float, constant): The Poisson's ratio used to construct the Lamé parameters
        positions (wp.vec3): The current position of the particle
        velocities (wp.vec3): The current velocity of the particle
        elastic_deformations (wp.mat33): The elastic part of the particle's current deformation gradient
        plastic_deformations (wp.mat33): The plastic part of the particle's current deformation gradient

    """

    volumes: wp.array[float]
    masses: wp.array[float]

    critical_compressions: wp.array[float]
    critical_stretches: wp.array[float]
    hardening_coefs: wp.array[float]
    mus: wp.array[float]
    lambdas: wp.array[float]

    positions: wp.array[wp.vec3]
    velocities: wp.array[wp.vec3]
    elastic_deformations: wp.array[wp.mat33]
    plastic_deformations: wp.array[wp.mat33]

    def init(self, i: int):
        self.volumes = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.masses = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.critical_compressions = wp.empty(
            shape=i, dtype=wp.float32, device="cuda"
        )
        self.critical_stretches = wp.empty(
            shape=i, dtype=wp.float32, device="cuda"
        )
        self.hardening_coefs = wp.empty(
            shape=i, dtype=wp.float32, device="cuda"
        )
        self.mus = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.lambdas = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.positions = wp.empty(shape=i, dtype=wp.vec3, device="cuda")
        self.velocities = wp.empty(shape=i, dtype=wp.vec3, device="cuda")
        self.elastic_deformations = wp.empty(
            shape=i, dtype=wp.mat33, device="cuda"
        )
        self.plastic_deformations = wp.empty(
            shape=i, dtype=wp.mat33, device="cuda"
        )

    def set_particles(self, plist: list[Particle]):
        self.init(len(plist))

        wp.launch(
            kernel=k_particles_fill_from_list,
            dim=len(plist),
            inputs=[wp.array(plist, dtype=Particle), self],
        )

        wp.launch(
            kernel=k_particles_fill_deformations, dim=len(self), inputs=[self]
        )

    def sample_cube(
            self,
            min_coord: wp.vec3,
            cell_size: wp.vec3,
            dimensions: wp.vec3,
            particles_per_cell: int = 8,
            density: float = 400.0,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0,
    ):
        num_particles = (
                int(dimensions[0])
                * int(dimensions[1])
                * int(dimensions[2])
                * particles_per_cell
        )

        particle_volume = (
                                  cell_size[0] * cell_size[1] * cell_size[2]
                          ) / particles_per_cell
        particle_mass = density * particle_volume

        self.init(num_particles)

        self.volumes.fill_(particle_volume)
        self.masses.fill_(particle_mass)
        self.critical_compressions.fill_(critical_compression)
        self.critical_stretches.fill_(critical_stretch)
        self.hardening_coefs.fill_(hardening_coef)
        self.mus.fill_(young_modulus / (2.0 * (1.0 + poisson_ratio)))
        self.lambdas.fill_(
            young_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )
        self.velocities.zero_()

        wp.launch(
            kernel=k_particles_sample_cube,
            dim=(
                dimensions[0],
                dimensions[1],
                dimensions[2],
                particles_per_cell,
            ),
            inputs=[
                min_coord,
                cell_size,
                dimensions[1],
                dimensions[2],
                particles_per_cell,
                seed,
                self.positions,
            ],
        )

        wp.launch(
            kernel=k_particles_fill_deformations, dim=len(self), inputs=[self]
        )

    def sample_sphere(
            self,
            center: wp.vec3,
            radius: float,
            particle_density: float,
            material_density: float = DEFAULT_DENSITY,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0,
    ):
        volume = (4.0 / 3.0) * math.pi * (radius ** 3)

        num_particles = max(1, int(round(particle_density * volume)))

        particle_volume = volume / float(num_particles)
        particle_mass = material_density * particle_volume

        self.init(num_particles)

        self.volumes.fill_(particle_volume)
        self.masses.fill_(particle_mass)
        self.critical_compressions.fill_(critical_compression)
        self.critical_stretches.fill_(critical_stretch)
        self.hardening_coefs.fill_(hardening_coef)
        self.mus.fill_(young_modulus / (2.0 * (1.0 + poisson_ratio)))
        self.lambdas.fill_(
            young_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )
        self.velocities.zero_()

        # 6. Sample spatial positions inside the sphere
        wp.launch(
            kernel=k_particles_sample_sphere,
            dim=num_particles,
            inputs=[center, radius, seed, self.positions],
        )

        # 7. Initialize deformation gradients
        wp.launch(
            kernel=k_particles_fill_deformations, dim=len(self), inputs=[self]
        )

    def sample_packed_snowball(
            self,
            center: wp.vec3,
            radius: float,
            particle_density: float,
            material_density: float = DEFAULT_DENSITY,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0,
            mass_outer_mult: float = 2.0,
            stiffness_outer_mult: float = 5.0,
            noise_amplitude: float = 0.4,
            noise_frequency: float = 20.0,
    ):
        volume = (4.0 / 3.0) * math.pi * (radius ** 3)

        num_particles = max(1, int(round(particle_density * volume)))
        particle_volume = volume / float(num_particles)

        self.init(num_particles)

        self.volumes.fill_(particle_volume)
        self.critical_compressions.fill_(critical_compression)
        self.critical_stretches.fill_(critical_stretch)
        self.hardening_coefs.fill_(hardening_coef)
        self.velocities.zero_()

        wp.launch(
            kernel=k_particles_sample_packed_snowball,
            dim=num_particles,
            inputs=[
                center,
                radius,
                seed,
                particle_volume,
                material_density,
                young_modulus,
                poisson_ratio,
                mass_outer_mult,
                stiffness_outer_mult,
                noise_amplitude,
                noise_frequency,
                self,
            ],
        )

        wp.launch(
            kernel=k_particles_fill_deformations, dim=len(self), inputs=[self]
        )

    def sample_mesh(
            self,
            mesh: wp.Mesh,
            particle_density: float,
            material_density: float = DEFAULT_DENSITY,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0,
            max_attempts: int = 1000,
    ):
        points = mesh.points.numpy()
        faces = mesh.indices.numpy().reshape(-1, 3)

        v0, v1, v2 = (
            points[faces[:, 0]],
            points[faces[:, 1]],
            points[faces[:, 2]],
        )
        volume = float(
            abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0
        )

        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        max_dist = float(np.linalg.norm(bbox_max - bbox_min))

        num_particles = max(1, int(round(particle_density * volume)))
        particle_volume = volume / float(num_particles)
        particle_mass = material_density * particle_volume

        self.init(num_particles)

        self.volumes.fill_(particle_volume)
        self.masses.fill_(particle_mass)
        self.critical_compressions.fill_(critical_compression)
        self.critical_stretches.fill_(critical_stretch)
        self.hardening_coefs.fill_(hardening_coef)
        self.mus.fill_(young_modulus / (2.0 * (1.0 + poisson_ratio)))
        self.lambdas.fill_(
            young_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )
        self.velocities.zero_()

        wp.launch(
            kernel=k_particles_sample_mesh,
            dim=num_particles,
            inputs=[
                mesh.id,
                wp.vec3(bbox_min),
                wp.vec3(bbox_max),
                max_dist,
                max_attempts,
                seed,
                self.positions,
            ],
        )

        wp.launch(
            kernel=k_particles_fill_deformations, dim=len(self), inputs=[self]
        )

    def _grow(self, extra: int, device=None) -> int:
        old = len(self)
        for name, dtype in _FIELDS.items():
            existing = getattr(self, name, None)
            dev = device
            if dev is None:
                dev = existing.device if existing is not None else "cuda"

            grown = wp.empty(shape=old + extra, dtype=dtype, device=dev)
            if old > 0 and existing is not None:
                wp.copy(grown, existing, 0, 0, old)
            setattr(self, name, grown)
        return old

    def add_snowball(
            self,
            center: wp.vec3,
            radius: float,
            initial_velocity: wp.vec3 = wp.vec3(0.0),
            spin_axis: wp.vec3 = wp.vec3(0.0, 1.0, 0.0),
            angular_speed: float = 0.0,  # rad/s, right-hand rule around spin_axis
            particle_diameter: float = DEFAULT_PARTICLE_DIAMETER,
            density: float = DEFAULT_DENSITY,  # noqa: F821  (already imported upstream)
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,  # noqa: F821
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,  # noqa: F821
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,  # noqa: F821
            young_modulus: float = DEFAULT_YOUNG_MODULUS,  # noqa: F821
            poisson_ratio: float = DEFAULT_POISSON_RATIO,  # noqa: F821
            shell_stiffness: float = 2.0,  # surface/core stiffness ratio (1.0 = uniform)
            shell_density: float = 1.25,  # surface/core density ratio (1.0 = uniform)
            noise_amplitude: float = 0.3,  # relative stiffness variation, keep in [0, 1)
            noise_scale: float = 4.0,  # noise features per radius
            seed: int = 5,
    ) -> int:
        center = np.asarray(center, dtype=np.float64)

        volume = 4.0 / 3.0 * np.pi * radius ** 3
        if volume < VOLUME_EPSILON:
            return 0

        count = int(volume / particle_diameter ** 3)
        if count == 0:
            return 0

        # Split the exact sphere volume evenly so total volume and mass are exact.
        particle_volume = volume / count

        existing = getattr(self, "volumes", None)
        if existing is None or len(existing) == 0:
            self.init(count)
            offset = 0
        else:
            offset = self._grow(count)
        device = self.positions.device

        initial_lambda, initial_mu = lame_parameters(young_modulus, poisson_ratio)

        axis = np.asarray(spin_axis, dtype=np.float64)
        axis_len = np.linalg.norm(axis)
        if axis_len > 0.0 and angular_speed != 0.0:
            omega = axis / axis_len * angular_speed
        else:
            omega = np.zeros(3)

        wp.launch(
            kernel=k_particles_sample_snowball,
            dim=count,
            inputs=[
                self,
                offset,
                count,
                wp.vec3(*center),
                float(radius),
                wp.vec3(*np.asarray(initial_velocity, dtype=np.float64)),
                wp.vec3(*omega),
                particle_volume,
                particle_volume * density,
                initial_lambda,
                initial_mu,
                critical_compression,
                critical_stretch,
                hardening_coef,
                shell_stiffness,
                shell_density,
                noise_amplitude,
                noise_scale,
                seed,
            ],
            device=device,
        )
        return count

    def __len__(self):
        return len(self.volumes)


@wp.kernel
def k_particles_fill_from_list(plist: wp.array[Particle], particles: Particles):
    i = wp.tid()
    p = plist[i]

    particles.volumes[i] = p.volume
    particles.masses[i] = p.mass
    particles.critical_compressions[i] = p.critical_compression
    particles.critical_stretches[i] = p.critical_stretch
    particles.hardening_coefs[i] = p.hardening_coef
    particles.mus[i] = p.initial_young_modulus / (2.0 * (1.0 + p.poisson_ratio))
    particles.lambdas[i] = (
            p.initial_young_modulus
            * p.poisson_ratio
            / ((1.0 + p.poisson_ratio) * (1.0 - 2.0 * p.poisson_ratio))
    )
    particles.positions[i] = p.position
    particles.velocities[i] = p.velocity
    particles.elastic_deformations[i] = p.elastic_deformation
    particles.plastic_deformations[i] = p.plastic_deformation


@wp.kernel
def k_particles_sample_cube(
        min_coord: wp.vec3,
        cell_size: wp.vec3,
        dim_y: int,
        dim_z: int,
        particles_per_cell: int,
        seed: wp.int32,
        positions: wp.array[wp.vec3],
):
    i, j, k, p = wp.tid()
    idx = ((i * dim_y + j) * dim_z + k) * particles_per_cell + p

    state = wp.rand_init(seed, idx)
    jitter = wp.vec3(
        wp.randf(state) * cell_size[0],
        wp.randf(state) * cell_size[1],
        wp.randf(state) * cell_size[2],
    )
    cell_origin = min_coord + wp.vec3(
        float(i) * cell_size[0],
        float(j) * cell_size[1],
        float(k) * cell_size[2],
    )

    positions[idx] = cell_origin + jitter


@wp.kernel
def k_particles_sample_sphere(
        center: wp.vec3,
        radius: float,
        seed: wp.int32,
        positions: wp.array[wp.vec3],
):
    idx = wp.tid()

    state = wp.rand_init(seed, idx)

    u = wp.randf(state)
    v = wp.randf(state)
    w = wp.randf(state)

    phi = 2.0 * 3.1415926535 * u
    cos_theta = 2.0 * v - 1.0
    sin_theta = wp.sqrt(1.0 - cos_theta * cos_theta)

    r = radius * wp.pow(w, 1.0 / 3.0)

    offset = wp.vec3(
        r * sin_theta * wp.cos(phi), r * sin_theta * wp.sin(phi), r * cos_theta
    )

    positions[idx] = center + offset


@wp.kernel
def k_particles_sample_packed_snowball(
        center: wp.vec3,
        radius: float,
        seed: wp.int32,
        particle_volume: float,
        base_density: float,
        base_young: float,
        poisson_ratio: float,
        mass_outer_mult: float,
        stiffness_outer_mult: float,
        noise_amplitude: float,
        noise_frequency: float,
        particles: Particles,
):
    idx = wp.tid()

    state = wp.rand_init(seed, idx)

    u = wp.randf(state)
    v = wp.randf(state)
    w = wp.randf(state)

    phi = 2.0 * 3.1415926535 * u
    cos_theta = 2.0 * v - 1.0
    sin_theta = wp.sqrt(1.0 - cos_theta * cos_theta)

    r = radius * wp.pow(w, 1.0 / 3.0)

    offset = wp.vec3(
        r * sin_theta * wp.cos(phi), r * sin_theta * wp.sin(phi), r * cos_theta
    )

    particles.positions[idx] = center + offset

    normalized_r = r / radius

    density_mult = wp.lerp(1.0, mass_outer_mult, normalized_r)
    local_density = base_density * density_mult
    particles.masses[idx] = local_density * particle_volume

    stiffness_mult = wp.lerp(1.0, stiffness_outer_mult, normalized_r)

    noise_seed = wp.uint32(seed)
    noise_val = wp.noise(noise_seed, offset * noise_frequency)

    local_young = (
            base_young * stiffness_mult * (1.0 + noise_amplitude * noise_val)
    )

    local_young = wp.max(0.01 * base_young, local_young)

    mu = local_young / (2.0 * (1.0 + poisson_ratio))
    lam = (
            local_young
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )

    particles.mus[idx] = mu
    particles.lambdas[idx] = lam


@wp.kernel
def k_particles_sample_mesh(
        mesh: wp.uint64,
        bbox_min: wp.vec3,
        bbox_max: wp.vec3,
        max_dist: float,
        max_attempts: int,
        seed: wp.int32,
        positions: wp.array[wp.vec3],
):
    idx = wp.tid()

    state = wp.rand_init(seed, idx)
    extent = bbox_max - bbox_min

    # rejection sampling: if max_attempts is exhausted, falls back to the
    # last candidate even if it landed outside the mesh
    pos = bbox_min
    for _attempt in range(max_attempts):
        candidate = wp.vec3(
            bbox_min[0] + wp.randf(state) * extent[0],
            bbox_min[1] + wp.randf(state) * extent[1],
            bbox_min[2] + wp.randf(state) * extent[2],
        )

        pos = candidate

        query = wp.mesh_query_point_sign_winding_number(
            mesh, candidate, max_dist
        )
        if query.result and query.sign < 0.0:
            break

    positions[idx] = pos


@wp.kernel
def k_particles_fill_deformations(particles: Particles):
    p = wp.tid()
    I = wp.identity(3, dtype=wp.float32)
    particles.elastic_deformations[p] = I
    particles.plastic_deformations[p] = I


@wp.kernel
def k_particles_sample_snowball(
        particles: Particles,  # noqa: F821
        offset: int,
        count: int,
        center: wp.vec3,
        radius: float,
        velocity: wp.vec3,
        angular_velocity: wp.vec3,
        volume: float,
        mass: float,
        initial_lambda: float,
        initial_mu: float,
        critical_compression: float,
        critical_stretch: float,
        hardening_coef: float,
        shell_stiffness: float,
        shell_density: float,
        noise_amplitude: float,
        noise_scale: float,
        seed: int,
):
    tid = wp.tid()
    if tid >= count:
        return

    state = wp.rand_init(seed, offset + tid)

    local = wp.sample_unit_sphere(state)
    position = center + radius * local

    shell = wp.length_sq(local)
    stiffness = (1.0 + (shell_stiffness - 1.0) * shell) / (1.0 + 0.6 * (shell_stiffness - 1.0))
    heaviness = (1.0 + (shell_density - 1.0) * shell) / (1.0 + 0.6 * (shell_density - 1.0))

    noise_state = wp.rand_init(seed, offset)
    n = wp.noise(noise_state, local * noise_scale)
    stiffness = stiffness * wp.max(1.0 + noise_amplitude * n, 0.1)

    i = offset + tid
    particles.positions[i] = position
    particles.velocities[i] = velocity + wp.cross(angular_velocity, position - center)
    particles.masses[i] = mass * heaviness
    particles.volumes[i] = volume
    particles.lambdas[i] = initial_lambda * stiffness
    particles.mus[i] = initial_mu * stiffness
    particles.critical_compressions[i] = critical_compression
    particles.critical_stretches[i] = critical_stretch
    particles.hardening_coefs[i] = hardening_coef
    particles.elastic_deformations[i] = wp.identity(n=3, dtype=float)
    particles.plastic_deformations[i] = wp.identity(n=3, dtype=float)


@wp.func
def hardening_factor(F_P: wp.mat33, xi: float) -> float:
    Jp = wp.determinant(F_P)
    return wp.exp(xi * (1.0 - Jp))


@wp.func
def mu_(
        plastic_deformation: wp.mat33, hardening_coef: float, initial_mu: float
) -> float:
    return initial_mu * hardening_factor(plastic_deformation, hardening_coef)


@wp.func
def lambda_(
        plastic_deformation: wp.mat33, hardening_coef: float, initial_lambda: float
) -> float:
    return initial_lambda * hardening_factor(plastic_deformation, hardening_coef)


def lame_parameters(young_modulus: float, poisson_ratio: float):
    lam = (
            young_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    mu = young_modulus / (2.0 + 2.0 * poisson_ratio)
    return lam, mu
