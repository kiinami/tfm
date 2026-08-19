import warp as wp

from mpm_explicit.constants import DEFAULT_CRITICAL_COMPRESSION, DEFAULT_CRITICAL_STRETCH, \
    DEFAULT_HARDENING_COEFFICIENT, DEFAULT_YOUNG_MODULUS, DEFAULT_POISSON_RATIO


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
        self.critical_compressions = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.critical_stretches = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.hardening_coefs = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.mus = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.lambdas = wp.empty(shape=i, dtype=wp.float32, device="cuda")
        self.positions = wp.empty(shape=i, dtype=wp.vec3, device="cuda")
        self.velocities = wp.empty(shape=i, dtype=wp.vec3, device="cuda")
        self.elastic_deformations = wp.empty(shape=i, dtype=wp.mat33, device="cuda")
        self.plastic_deformations = wp.empty(shape=i, dtype=wp.mat33, device="cuda")

    def set_particles(self, plist: list[Particle]):
        self.init(len(plist))

        wp.launch(
            kernel=k_fill_particles,
            dim=len(plist),
            inputs=[wp.array(plist, dtype=Particle), self],
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
            seed: int = 0
    ):
        num_particles = int(dimensions[0]) * int(dimensions[1]) * int(dimensions[2]) * particles_per_cell

        particle_volume = (cell_size[0] * cell_size[1] * cell_size[2]) / particles_per_cell
        particle_mass = density * particle_volume

        self.init(num_particles)

        self.volumes.fill_(particle_volume)
        self.masses.fill_(particle_mass)
        self.critical_compressions.fill_(critical_compression)
        self.critical_stretches.fill_(critical_stretch)
        self.hardening_coefs.fill_(hardening_coef)
        self.mus.fill_(young_modulus / (2.0 * (1.0 + poisson_ratio)))
        self.lambdas.fill_(young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)))

        self.velocities.zero_()
        self.elastic_deformations.fill_(wp.mat33(1.0))
        self.plastic_deformations.fill_(wp.mat33(1.0))

        wp.launch(
            kernel=k_sample_cube,
            dim=(dimensions[0], dimensions[1], dimensions[2], particles_per_cell),
            inputs=[min_coord, cell_size, dimensions[1], dimensions[2], particles_per_cell, seed, self.positions],
        )

    def __len__(self):
        return len(self.volumes)


@wp.kernel
def k_fill_particles(plist: wp.array[Particle], particles: Particles):
    i = wp.tid()
    p = plist[i]

    particles.volumes[i] = p.volume
    particles.masses[i] = p.mass
    particles.critical_compressions[i] = p.critical_compression
    particles.critical_stretches[i] = p.critical_stretch
    particles.hardening_coefs[i] = p.hardening_coef
    particles.mus[i] = p.initial_young_modulus / (2.0 * (1.0 + p.poisson_ratio))
    particles.lambdas[i] = p.initial_young_modulus * p.poisson_ratio / ((1.0 + p.poisson_ratio) * (1.0 - 2.0 * p.poisson_ratio))
    particles.positions[i] = p.position
    particles.velocities[i] = p.velocity
    particles.elastic_deformations[i] = p.elastic_deformation
    particles.plastic_deformations[i] = p.plastic_deformation


@wp.kernel
def k_sample_cube(
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
    jitter = wp.vec3(wp.randf(state) * cell_size[0], wp.randf(state) * cell_size[1], wp.randf(state) * cell_size[2])
    cell_origin = min_coord + wp.vec3(float(i) * cell_size[0], float(j) * cell_size[1], float(k) * cell_size[2])

    positions[idx] = cell_origin + jitter


@wp.func
def mu_(plastic_deformation: wp.mat33, hardening_coef: float, initial_mu: float) -> float:
    Jp = wp.determinant(plastic_deformation)
    return initial_mu * wp.exp(hardening_coef * (1.0 - Jp))


@wp.func
def lambda_(plastic_deformation: wp.mat33, hardening_coef: float, initial_lambda: float) -> float:
    Jp = wp.determinant(plastic_deformation)
    return initial_lambda * wp.exp(hardening_coef * (1.0 - Jp))
