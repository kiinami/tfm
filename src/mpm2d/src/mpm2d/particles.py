import numpy as np
import warp as wp

from mpm2d.constants import (
    AREA_EPSILON,
    DEFAULT_CRITICAL_COMPRESSION,
    DEFAULT_CRITICAL_STRETCH,
    DEFAULT_DENSITY,
    DEFAULT_HARDENING_COEFFICIENT,
    DEFAULT_PARTICLE_DIAMETER,
    DEFAULT_POISSON_RATIO,
    DEFAULT_YOUNG_MODULUS,
    MAX_SAMPLE_TRIES,
)
from mpm2d.utils import polygon_area, regular_polygon

_FIELDS = {
    "volumes": wp.float32,
    "masses": wp.float32,
    "critical_compressions": wp.float32,
    "critical_stretches": wp.float32,
    "hardening_coefs": wp.float32,
    "mus": wp.float32,
    "lambdas": wp.float32,
    "positions": wp.vec2,
    "velocities": wp.vec2,
    "elastic_deformations": wp.mat22,
    "plastic_deformations": wp.mat22,
}


@wp.struct
class Particles:
    volumes: wp.array[float]
    masses: wp.array[float]

    critical_compressions: wp.array[float]
    critical_stretches: wp.array[float]
    hardening_coefs: wp.array[float]
    mus: wp.array[float]
    lambdas: wp.array[float]

    positions: wp.array[wp.vec2]
    velocities: wp.array[wp.vec2]
    elastic_deformations: wp.array[wp.mat22]
    plastic_deformations: wp.array[wp.mat22]

    def init(self, i: int):
        self.volumes = wp.empty(shape=i, dtype=wp.float32)
        self.masses = wp.empty(shape=i, dtype=wp.float32)
        self.critical_compressions = wp.empty(shape=i, dtype=wp.float32)
        self.critical_stretches = wp.empty(shape=i, dtype=wp.float32)
        self.hardening_coefs = wp.empty(shape=i, dtype=wp.float32)
        self.mus = wp.empty(shape=i, dtype=wp.float32)
        self.lambdas = wp.empty(shape=i, dtype=wp.float32)
        self.positions = wp.empty(shape=i, dtype=wp.vec2)
        self.velocities = wp.empty(shape=i, dtype=wp.vec2)
        self.elastic_deformations = wp.empty(shape=i, dtype=wp.mat22)
        self.plastic_deformations = wp.empty(shape=i, dtype=wp.mat22)

    def __len__(self):
        return len(self.volumes)

    def _grow(self, extra: int, device=None) -> int:
        old = len(self)
        for name, dtype in _FIELDS.items():
            grown = wp.empty(shape=old + extra, dtype=dtype, device=device)
            existing = getattr(self, name)
            if old > 0 and existing is not None:
                wp.copy(grown, existing, 0, 0, old)
            setattr(self, name, grown)
        return old

    def add_snowball(
        self,
        center: wp.vec2[float],
        radius: float,
        initial_velocity: wp.vec2[float] = wp.vec2(0.0),
        segments: int = 8,
        particle_diameter: float = DEFAULT_PARTICLE_DIAMETER,
        density: float = DEFAULT_DENSITY,
        critical_compression: float = DEFAULT_CRITICAL_COMPRESSION,
        critical_stretch: float = DEFAULT_CRITICAL_STRETCH,
        hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT,
        young_modulus: float = DEFAULT_YOUNG_MODULUS,
        poisson_ratio: float = DEFAULT_POISSON_RATIO,
        seed: int = 5,
    ):
        center = np.asarray(center, dtype=np.float64)
        vertices = regular_polygon(center, radius, segments)

        area = polygon_area(vertices)
        if area < AREA_EPSILON or segments < 3:
            return 0

        particle_area = particle_diameter * particle_diameter
        count = int(area / particle_area)
        if count == 0:
            return 0

        if not self.volumes:
            self.init(count)
            offset = 0
        else:
            offset = self._grow(count)

        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        bbox_center = 0.5 * (lower + upper)

        wp.launch(
            k_particles_sample_snowball,
            dim=count,
            inputs=[
                self,
                offset,
                count,
                wp.array(vertices, dtype=wp.vec2),
                wp.vec2(*lower),
                wp.vec2(*upper),
                wp.vec2(*bbox_center),
                wp.vec2(*(upper - bbox_center)),
                wp.vec2(*np.asarray(initial_velocity, dtype=np.float64)),
                particle_area * density,
                *lame_parameters(young_modulus, poisson_ratio),
                critical_compression,
                critical_stretch,
                hardening_coef,
                seed,
            ],
        )


def lame_parameters(young_modulus: float, poisson_ratio: float):
    lam = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    mu = young_modulus / (2.0 + 2.0 * poisson_ratio)
    return lam, mu


@wp.func
def polygon_contains(vertices: wp.array[wp.vec2], p: wp.vec2) -> bool:
    n = vertices.shape[0]
    crossings = wp.int32(0)

    for i in range(n):
        vi = vertices[i]
        vj = vertices[(i + n - 1) % n]

        above_i = vi[1] > p[1]
        above_j = vj[1] > p[1]
        crosses = wp.bool(False)
        if above_i != above_j:
            edge_x = (vj[0] - vi[0]) * (p[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]
            crosses = p[0] < edge_x
        crossings = crossings + int(crosses)

    return (crossings % 2) == 1


@wp.kernel
def k_particles_sample_snowball(
    particles: Particles,
    offset: int,
    count: int,
    vertices: wp.array[wp.vec2],
    lower: wp.vec2,
    upper: wp.vec2,
    center: wp.vec2,
    half_extent: wp.vec2,
    velocity: wp.vec2,
    mass: float,
    initial_lambda: float,
    initial_mu: float,
    critical_compression: float,
    critical_stretch: float,
    max_hardening: float,
    seed: int,
):
    tid = wp.tid()
    if tid >= count:
        return

    state = wp.rand_init(seed, tid)

    position = center
    for _ in range(MAX_SAMPLE_TRIES):
        candidate = wp.vec2(
            wp.randf(state, lower[0], upper[0]),
            wp.randf(state, lower[1], upper[1]),
        )
        if polygon_contains(vertices, candidate):
            position = candidate
            break

    i = offset + tid
    particles.positions[i] = position
    particles.velocities[i] = velocity
    particles.masses[i] = mass
    particles.volumes[i] = 0.0
    particles.lambdas[i] = initial_lambda
    particles.mus[i] = initial_mu
    particles.critical_compressions[i] = critical_compression
    particles.critical_stretches[i] = critical_stretch
    particles.hardening_coefs[i] = max_hardening
    particles.elastic_deformations[i] = wp.identity(n=2, dtype=float)
    particles.plastic_deformations[i] = wp.identity(n=2, dtype=float)


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
