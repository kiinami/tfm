from pathlib import Path

import numpy as np
import trimesh
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
def safe_svd3(M: wp.mat33):
    U = wp.mat33(1.0)
    V = wp.mat33(1.0)
    sigma = wp.vec3(0.0)

    wp.svd3(M, U, sigma, V)

    if wp.determinant(U) < 0.0:
        U = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2],
        )
        sigma[2] = -sigma[2]

    if wp.determinant(V) < 0.0:
        V = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2],
        )
        sigma[2] = -sigma[2]

    return U, sigma, V


@wp.func
def extract_rotation(F: wp.mat33) -> wp.mat33:
    U, sigma, V = safe_svd3(F)

    return U @ wp.transpose(V)


@wp.func
def polar_decompose(F: wp.mat33) -> tuple[wp.mat33, wp.mat33]:
    U, sigma, V = safe_svd3(F)
    R = U @ wp.transpose(V)
    S = V @ wp.diag(sigma) @ wp.transpose(V)
    return R, S


@wp.func
def axial(W: wp.mat33) -> wp.vec3:
    return wp.vec3(W[2, 1], W[0, 2], W[1, 0])


@wp.func
def skew(w: wp.vec3) -> wp.mat33:
    return wp.mat33(
        0.0, -w[2], w[1],
        w[2], 0.0, -w[0],
        -w[1], w[0], 0.0,
    )


@wp.func
def rotation_differential(dF: wp.mat33, R: wp.mat33, S: wp.mat33) -> wp.mat33:
    skew_rhs = wp.transpose(R) @ dF - wp.transpose(dF) @ R
    rhs = axial(skew_rhs)

    trace_S = S[0, 0] + S[1, 1] + S[2, 2]
    I = wp.identity(3, dtype=wp.float32)
    M = trace_S * I - S

    w = wp.inverse(M) @ rhs
    return R @ skew(w)


@wp.func
def cofactor(A: wp.mat33) -> wp.mat33:
    c00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    c01 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    c02 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]

    c10 = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    c11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    c12 = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]

    c20 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    c21 = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    c22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    return wp.mat33(
        c00, c01, c02,
        c10, c11, c12,
        c20, c21, c22
    )


@wp.func
def cofactor_differential(F: wp.mat33, dF: wp.mat33) -> wp.mat33:
    c00 = dF[1, 1] * F[2, 2] + F[1, 1] * dF[2, 2] - dF[1, 2] * F[2, 1] - F[1, 2] * dF[2, 1]
    c01 = dF[1, 2] * F[2, 0] + F[1, 2] * dF[2, 0] - dF[1, 0] * F[2, 2] - F[1, 0] * dF[2, 2]
    c02 = dF[1, 0] * F[2, 1] + F[1, 0] * dF[2, 1] - dF[1, 1] * F[2, 0] - F[1, 1] * dF[2, 0]

    c10 = dF[0, 2] * F[2, 1] + F[0, 2] * dF[2, 1] - dF[0, 1] * F[2, 2] - F[0, 1] * dF[2, 2]
    c11 = dF[0, 0] * F[2, 2] + F[0, 0] * dF[2, 2] - dF[0, 2] * F[2, 0] - F[0, 2] * dF[2, 0]
    c12 = dF[0, 1] * F[2, 0] + F[0, 1] * dF[2, 0] - dF[0, 0] * F[2, 1] - F[0, 0] * dF[2, 1]

    c20 = dF[0, 1] * F[1, 2] + F[0, 1] * dF[1, 2] - dF[0, 2] * F[1, 1] - F[0, 2] * dF[1, 1]
    c21 = dF[0, 2] * F[1, 0] + F[0, 2] * dF[1, 0] - dF[0, 0] * F[1, 2] - F[0, 0] * dF[1, 2]
    c22 = dF[0, 0] * F[1, 1] + F[0, 0] * dF[1, 1] - dF[0, 1] * F[1, 0] - F[0, 1] * dF[1, 0]

    return wp.mat33(c00, c01, c02, c10, c11, c12, c20, c21, c22)


def import_mesh(file: Path | str) -> wp.Mesh:
    tm = trimesh.load_mesh(file)
    rotation = trimesh.transformations.rotation_matrix(
        np.radians(90), [1, 0, 0]
    )
    tm.apply_transform(rotation)
    points = np.asarray(tm.vertices, dtype=np.float32)
    indices = np.asarray(tm.faces, dtype=np.int32)
    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device="cuda"),
        indices=wp.array(indices.reshape(-1), dtype=wp.int32, device="cuda"),
    )
    return mesh
