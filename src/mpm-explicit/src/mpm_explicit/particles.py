import math

import numpy as np
import warp as wp

from mpm_explicit.constants import (
    DEFAULT_CRITICAL_COMPRESSION,
    DEFAULT_CRITICAL_STRETCH,
    DEFAULT_DENSITY,
    DEFAULT_HARDENING_COEFFICIENT,
    DEFAULT_POISSON_RATIO,
    DEFAULT_YOUNG_MODULUS,
)


def sphere_particle_count(radius: float, particle_diam: float) -> int:
    volume = (4.0 / 3.0) * math.pi * (radius ** 3)
    particle_volume = particle_diam ** 3
    return int(volume / particle_volume)


@wp.struct
class Particles:
    volumes: wp.array[float]
    masses: wp.array[float]
    densities: wp.array[float]

    critical_compressions: wp.array[float]
    critical_stretches: wp.array[float]
    hardening_coefs: wp.array[float]
    mus: wp.array[float]
    lambdas: wp.array[float]

    positions: wp.array[wp.vec3]
    velocities: wp.array[wp.vec3]
    elastic_deformations: wp.array[wp.mat33]
    plastic_deformations: wp.array[wp.mat33]

    def init(self, n: int):
        self.volumes = wp.empty(shape=n, dtype=wp.float32)
        self.masses = wp.empty(shape=n, dtype=wp.float32)
        self.densities = wp.empty(shape=n, dtype=wp.float32)

        self.critical_compressions = wp.empty(shape=n, dtype=wp.float32)
        self.critical_stretches = wp.empty(shape=n, dtype=wp.float32)
        self.hardening_coefs = wp.empty(shape=n, dtype=wp.float32)
        self.mus = wp.empty(shape=n, dtype=wp.float32)
        self.lambdas = wp.empty(shape=n, dtype=wp.float32)

        self.positions = wp.empty(shape=n, dtype=wp.vec3)
        self.velocities = wp.empty(shape=n, dtype=wp.vec3)
        self.elastic_deformations = wp.empty(shape=n, dtype=wp.mat33)
        self.plastic_deformations = wp.empty(shape=n, dtype=wp.mat33)

    def fill_deformations(self):
        wp.launch(
            kernel=k_particles_fill_deformations,
            dim=len(self),
            inputs=[self]
        )

    def sample_sphere(
            self,
            center: wp.vec3,
            radius: float,
            particle_diam: float,
            velocity: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
            material_density: float = DEFAULT_DENSITY,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0,
    ):
        num_particles = sphere_particle_count(radius, particle_diam)

        self.init(num_particles)

        self.fill_sphere(
            offset=0,
            center=center,
            radius=radius,
            particle_diam=particle_diam,
            velocity=velocity,
            material_density=material_density,
            critical_compression=critical_compression,
            critical_stretch=critical_stretch,
            hardening_coef=hardening_coef,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            seed=seed,
        )

        self.fill_deformations()

    def fill_sphere(
            self,
            offset: int,
            center: wp.vec3,
            radius: float,
            particle_diam: float,
            velocity: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
            material_density: float = DEFAULT_DENSITY,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0
    ) -> int:
        num_particles = sphere_particle_count(radius, particle_diam)
        end = offset + num_particles

        particle_volume = particle_diam ** 3
        particle_mass = material_density * particle_volume
        lambda0 = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        mu0 = young_modulus / (2.0 * (1.0 + poisson_ratio))

        self.masses[offset:end].fill_(particle_mass)
        self.volumes[offset:end].fill_(particle_volume)
        self.velocities[offset:end].fill_(velocity)

        self.mus[offset:end].fill_(mu0)
        self.lambdas[offset:end].fill_(lambda0)
        self.critical_compressions[offset:end].fill_(critical_compression)
        self.critical_stretches[offset:end].fill_(critical_stretch)
        self.hardening_coefs[offset:end].fill_(hardening_coef)

        wp.launch(
            kernel=k_particles_sample_sphere,
            dim=num_particles,
            inputs=[center, radius, seed, self.positions[offset:end]],
        )

        return num_particles

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

    def __len__(self):
        return len(self.volumes)


@wp.kernel
def k_particles_sample_sphere(
        center: wp.vec3,
        radius: float,
        seed: wp.int32,
        positions: wp.array[wp.vec3],
):
    p = wp.tid()

    state = wp.rand_init(seed, p)

    u = wp.randf(state)
    v = wp.randf(state)
    w = wp.randf(state)

    phi = 2.0 * 3.1415926535 * u
    cos_theta = 2.0 * v - 1.0
    sin_theta = wp.sqrt(1.0 - cos_theta * cos_theta)
    r = radius * wp.pow(w, 1.0 / 3.0)

    offset = wp.vec3(
        r * sin_theta * wp.cos(phi),
        r * sin_theta * wp.sin(phi),
        r * cos_theta
    )

    positions[p] = center + offset


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
