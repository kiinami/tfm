import numpy as np
import warp as wp

from mpm_implicit.utils import cofactor, cofactor_differential, polar_decompose, rotation_differential

wp.init()

MU = 1.0
LAMBDA = 1.0


@wp.func
def piola_stress(F: wp.mat33) -> wp.mat33:
    R, S = polar_decompose(F)
    J = wp.determinant(F)
    return 2.0 * MU * (F - R) + LAMBDA * (J - 1.0) * cofactor(F)


@wp.func
def piola_stress_differential(F: wp.mat33, dF: wp.mat33) -> wp.mat33:
    R, S = polar_decompose(F)
    dR = rotation_differential(dF, R, S)
    cof_F = cofactor(F)
    dcof_F = cofactor_differential(F, dF)

    contraction = float(0.0)
    for r in range(3):
        for c in range(3):
            contraction += cof_F[r, c] * dF[r, c]

    J = wp.determinant(F)

    return (
            2.0 * MU * (dF - dR)
            + LAMBDA * contraction * cof_F
            + LAMBDA * (J - 1.0) * dcof_F
    )


@wp.kernel
def k_check(F_in: wp.array(dtype=wp.mat33), dF_in: wp.array(dtype=wp.mat33), eps: float, out: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    F = F_in[tid]
    dF = dF_in[tid]

    p_plus = piola_stress(F + eps * dF)
    p = piola_stress(F)
    finite_diff = (p_plus - p) / eps

    analytic = piola_stress_differential(F, dF)

    out[tid] = finite_diff - analytic


rng = np.random.default_rng(0)
n = 20
Fs = np.stack([np.eye(3) + 0.3 * rng.standard_normal((3, 3)) for _ in range(n)]).astype(np.float32)
dFs = (0.01 * rng.standard_normal((n, 3, 3))).astype(np.float32)

F_arr = wp.array(Fs, dtype=wp.mat33)
dF_arr = wp.array(dFs, dtype=wp.mat33)
out_arr = wp.zeros(n, dtype=wp.mat33)

for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]:
    out_arr = wp.zeros(n, dtype=wp.mat33)
    wp.launch(k_check, dim=n, inputs=[F_arr, dF_arr, eps, out_arr])
    errors = np.abs(out_arr.numpy()).max(axis=(1, 2))
    print(f"eps={eps:.0e}  worst case: {errors.max():.6e}")
