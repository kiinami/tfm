import warp as wp

from mpm_explicit.constants import DEFAULT_CRITICAL_COMPRESSION, DEFAULT_CRITICAL_STRETCH, \
    DEFAULT_HARDENING_COEFFICIENT, DEFAULT_YOUNG_MODULUS, DEFAULT_POISSON_RATIO


@wp.struct
class Particle:
    """
    Structure containing the data for a particle

    Attributes:
        volume (float, constant): The initial volume of the particle, assigned on the first iteration
        mass (float, constant): The mass of the particle
        critical_compression (float, constant): The compression threshold
        critical_stretch (float, constant): The stretch threshold
        hardening_coef (float, constant): A coefficient that defines how fast the material breaks once yielding
        initial_young_modulus (float, constant): The overall stiffness of the material
        poisson_ratio (float, constant): The Poisson's ratio used to construct the Lamé parameters
        position (wp.vec3): The current position of the particle
        velocity (wp.vec3): The current velocity of the particle
        elastic_deformation (wp.mat33): The elastic part of the particle's current deformation gradient
        plastic_deformation (wp.mat33): The plastic part of the particle's current deformation gradient
    """
    volume: float
    mass: float

    critical_compression: float = DEFAULT_CRITICAL_COMPRESSION
    critical_stretch: float = DEFAULT_CRITICAL_STRETCH
    hardening_coef: float = DEFAULT_HARDENING_COEFFICIENT
    initial_young_modulus: float = DEFAULT_YOUNG_MODULUS
    poisson_ratio: float = DEFAULT_POISSON_RATIO

    position: wp.vec3
    velocity: wp.vec3
    elastic_deformation: wp.mat33
    plastic_deformation: wp.mat33
