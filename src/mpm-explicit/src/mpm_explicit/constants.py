import warp as wp

DEFAULT_DT = wp.constant(0.00001)
DEFAULT_DENSITY = wp.constant(400.0)
DEFAULT_CRITICAL_COMPRESSION = wp.constant(2.5e-2)
DEFAULT_CRITICAL_STRETCH = wp.constant(7.5e-3)
DEFAULT_HARDENING_COEFFICIENT = wp.constant(10)
DEFAULT_YOUNG_MODULUS = wp.constant(1.4e5)
DEFAULT_POISSON_RATIO =  wp.constant(0.2)

EPSILON = wp.constant(1e-12)
GRAVITY = wp.constant(wp.vec3(0.0, 0.0, -9.81))
COULOMB_FRICTION = wp.constant(0.5)
PICFLIP_ALPHA = wp.constant(0.95)
MAX_COLLISION_DIST = wp.constant(100.0)
