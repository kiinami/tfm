import numpy as np
import warp as wp


def regular_polygon(center, radius: float, segments: int) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(segments) / segments
    return np.stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
        ],
        axis=1,
    )


def polygon_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    xj, yj = np.roll(x, 1), np.roll(y, 1)
    return float(abs(np.sum((xj + x) * (yj - y)) / 2.0))


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
def safe_svd2(M: wp.mat22):
    U = wp.mat22(1.0)
    V = wp.mat22(1.0)
    sigma = wp.vec2(0.0)

    wp.svd2(M, U, sigma, V)

    if wp.determinant(U) < 0.0:
        U = wp.mat22(
            U[0, 0], -U[0, 1],
            U[1, 0], -U[1, 1],
        )
        sigma[1] = -sigma[1]

    if wp.determinant(V) < 0.0:
        V = wp.mat22(
            V[0, 0], -V[0, 1],
            V[1, 0], -V[1, 1],
        )
        sigma[1] = -sigma[1]

    return U, sigma, V


@wp.func
def extract_rotation(F: wp.mat22) -> wp.mat22:
    U, sigma, V = safe_svd2(F)

    return U @ wp.transpose(V)


@wp.func
def cofactor(A: wp.mat22) -> wp.mat22:
    c00 = A[1, 1]
    c01 = -A[1, 0]

    c10 = -A[0, 1]
    c11 = A[0, 0]

    return wp.mat22(
        c00, c01,
        c10, c11
    )
