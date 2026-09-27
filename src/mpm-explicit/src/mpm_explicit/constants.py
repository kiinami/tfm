import warp as wp

DEFAULT_DENSITY = wp.constant(400.0)
DEFAULT_CRITICAL_COMPRESSION = wp.constant(2.5e-2)
DEFAULT_CRITICAL_STRETCH = wp.constant(7.5e-3)
DEFAULT_HARDENING_COEFFICIENT = wp.constant(10)
DEFAULT_YOUNG_MODULUS = wp.constant(1.4e5)
DEFAULT_POISSON_RATIO = wp.constant(0.2)

DEFAULT_CFL = wp.constant(0.4)
DEFAULT_MIN_DT = wp.constant(1e-6)
DEFAULT_MAX_DT = wp.constant(5e-4)

EPSILON = wp.constant(1e-8)
EPSILON_SQ = wp.constant(1e-9)
GRAVITY = wp.constant(wp.vec3(0.0, 0.0, -9.81))
COULOMB_FRICTION = wp.constant(0.9)
PICFLIP_ALPHA = wp.constant(0.95)
MAX_COLLISION_DIST = wp.constant(0.02)
