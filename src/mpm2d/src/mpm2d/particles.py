import math

import warp as wp

from mpm2d.constants import (
    DEFAULT_CRITICAL_COMPRESSION,
    DEFAULT_CRITICAL_STRETCH,
    DEFAULT_DENSITY,
    DEFAULT_HARDENING_COEFFICIENT,
    DEFAULT_POISSON_RATIO,
    DEFAULT_YOUNG_MODULUS,
)
from mpm2d.utils import Polygon, polygon_area, polygon_contains


def circle_particle_count(radius: float, particle_diam: float) -> int:
    area = math.pi * (radius ** 2)
    particle_area = particle_diam ** 2
    return int(area / particle_area)


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

    positions: wp.array[wp.vec2]
    velocities: wp.array[wp.vec2]
    elastic_deformations: wp.array[wp.mat22]
    plastic_deformations: wp.array[wp.mat22]

    def init(self, n: int):
        self.volumes = wp.empty(shape=n, dtype=wp.float32)
        self.masses = wp.empty(shape=n, dtype=wp.float32)
        self.densities = wp.empty(shape=n, dtype=wp.float32)

        self.critical_compressions = wp.empty(shape=n, dtype=wp.float32)
        self.critical_stretches = wp.empty(shape=n, dtype=wp.float32)
        self.hardening_coefs = wp.empty(shape=n, dtype=wp.float32)
        self.mus = wp.empty(shape=n, dtype=wp.float32)
        self.lambdas = wp.empty(shape=n, dtype=wp.float32)

        self.positions = wp.empty(shape=n, dtype=wp.vec2)
        self.velocities = wp.empty(shape=n, dtype=wp.vec2)
        self.elastic_deformations = wp.empty(shape=n, dtype=wp.mat22)
        self.plastic_deformations = wp.empty(shape=n, dtype=wp.mat22)

    def fill_deformations(self):
        wp.launch(
            kernel=k_particles_fill_deformations,
            dim=len(self),
            inputs=[self]
        )

    def sample_circle(
            self,
            center: wp.vec2,
            radius: float,
            particle_diam: float,
            velocity: wp.vec2 = wp.vec2(0.0, 0.0),
            material_density: float = DEFAULT_DENSITY,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0,
    ):
        num_particles = circle_particle_count(radius, particle_diam)

        self.init(num_particles)

        self.fill_circle(
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

    def fill_circle(
            self,
            offset: int,
            center: wp.vec2,
            radius: float,
            particle_diam: float,
            velocity: wp.vec2 = wp.vec2(0.0, 0.0),
            material_density: float = DEFAULT_DENSITY,
            critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
            critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
            hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
            young_modulus: float = DEFAULT_YOUNG_MODULUS,
            poisson_ratio: float = DEFAULT_POISSON_RATIO,
            seed: int = 0
    ) -> int:
        num_particles = circle_particle_count(radius, particle_diam)
        end = offset + num_particles

        particle_volume = particle_diam ** 2
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
            kernel=k_particles_sample_circle,
            dim=num_particles,
            inputs=[center, radius, seed, self.positions[offset:end]],
        )

        return num_particles

    def sample_polygon(
            self,
            polygon: Polygon,
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
        points = polygon.points.numpy()

        area = polygon_area(points)

        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        num_particles = max(1, int(round(particle_density * area)))
        particle_volume = area / float(num_particles)
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
            kernel=k_particles_sample_polygon,
            dim=num_particles,
            inputs=[
                polygon.points,
                wp.vec2(bbox_min),
                wp.vec2(bbox_max),
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
def k_particles_sample_circle(
        center: wp.vec2,
        radius: float,
        seed: wp.int32,
        positions: wp.array[wp.vec2],
):
    p = wp.tid()

    state = wp.rand_init(seed, p)

    u = wp.randf(state)
    w = wp.randf(state)

    phi = 2.0 * 3.1415926535 * u
    r = radius * wp.sqrt(w)

    offset = wp.vec2(
        r * wp.cos(phi),
        r * wp.sin(phi),
    )

    positions[p] = center + offset


@wp.kernel
def k_particles_sample_polygon(
        vertices: wp.array[wp.vec2],
        bbox_min: wp.vec2,
        bbox_max: wp.vec2,
        max_attempts: int,
        seed: wp.int32,
        positions: wp.array[wp.vec2],
):
    idx = wp.tid()

    state = wp.rand_init(seed, idx)
    extent = bbox_max - bbox_min

    pos = bbox_min
    for _attempt in range(max_attempts):
        candidate = wp.vec2(
            bbox_min[0] + wp.randf(state) * extent[0],
            bbox_min[1] + wp.randf(state) * extent[1],
        )

        pos = candidate

        if polygon_contains(vertices, 0, vertices.shape[0], candidate):
            break

    positions[idx] = pos


@wp.kernel
def k_particles_fill_deformations(particles: Particles):
    p = wp.tid()
    I = wp.identity(2, dtype=wp.float32)
    particles.elastic_deformations[p] = I
    particles.plastic_deformations[p] = I
