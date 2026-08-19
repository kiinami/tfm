import warp as wp


@wp.func
def bspline_w(x: float) -> float:
    x = wp.abs(x)
    if x < 1.0:
        return 0.5 * x * x * x - x * x + 2.0 / 3.0
    elif x < 2.0:
        return -1.0 / 6.0 * x * x * x + x * x - 2.0 * x + 4.0 / 3.0
    return 0.0


@wp.func
def bspline_dw(x: float) -> float:
    x_abs = wp.abs(x)
    if x_abs < 1.0:
        return 1.5 * x_abs * x - 2.0 * x
    elif x_abs < 2.0:
        return -0.5 * x_abs * x + 2.0 * x - 2.0 * wp.sign(x)
    return 0.0


@wp.func
def extract_rotation(F: wp.mat33) -> wp.mat33:
    """
    Compute the polar decomposition of a 3x3 matrix F into a rotation R and a symmetric matrix S such that F = R * S, then return the rotation.
    """
    U = wp.mat33(1.0)
    V = wp.mat33(1.0)
    sigma = wp.vec3(0.0)

    wp.svd3(F, U, sigma, V)

    R = U @ wp.transpose(V)

    return R
