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
        return -0.5 * x_abs * x + 2.0 * x
    return 0.0
