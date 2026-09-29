import numpy as np
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
def polar_decompose(F: wp.mat22) -> tuple[wp.mat22, wp.mat22]:
    U, sigma, V = safe_svd2(F)
    R = U @ wp.transpose(V)
    S = V @ wp.diag(sigma) @ wp.transpose(V)
    return R, S


@wp.func
def axial(W: wp.mat22) -> float:
    return W[1, 0]


@wp.func
def skew(w: float) -> wp.mat22:
    return wp.mat22(
        0.0, -w,
        w, 0.0,
    )


@wp.func
def rotation_differential(dF: wp.mat22, R: wp.mat22, S: wp.mat22) -> wp.mat22:
    # in 2D, J S + S J = tr(S) J for any symmetric S, so the 3D linear system
    # (tr(S) I - S) w = axial(...) collapses to a scalar division
    skew_rhs = wp.transpose(R) @ dF - wp.transpose(dF) @ R
    rhs = axial(skew_rhs)

    trace_S = S[0, 0] + S[1, 1]

    w = rhs / trace_S
    return R @ skew(w)


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


@wp.func
def cofactor_differential(F: wp.mat22, dF: wp.mat22) -> wp.mat22:
    # the 2D cofactor is linear in its argument
    return cofactor(dF)


class Polygon:
    """
    A closed 2D polygon obstacle, with its vertices in counter-clockwise order
    """
    points: wp.array[wp.vec2]

    def __init__(self, points: np.ndarray):
        points = np.asarray(points, dtype=np.float32)
        if _signed_area(points) < 0.0:
            points = points[::-1].copy()
        self.points = wp.array(points, dtype=wp.vec2)


def _signed_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    xj, yj = np.roll(x, -1), np.roll(y, -1)
    return float(np.sum(x * yj - xj * y) / 2.0)


def polygon_area(vertices: np.ndarray) -> float:
    return abs(_signed_area(vertices))


def regular_polygon(center, radius: float, segments: int) -> Polygon:
    theta = 2.0 * np.pi * np.arange(segments) / segments
    return Polygon(
        np.stack(
            [
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
            ],
            axis=1,
        )
    )


def box_polygon(lower, upper) -> Polygon:
    return Polygon(
        np.array(
            [
                [lower[0], lower[1]],
                [upper[0], lower[1]],
                [upper[0], upper[1]],
                [lower[0], upper[1]],
            ]
        )
    )


@wp.struct
class Obstacles:
    """
    A set of polygons packed for use in kernels. The vertices of polygon b are
    vertices[offsets[b]:offsets[b + 1]]
    """
    vertices: wp.array[wp.vec2]
    offsets: wp.array[int]

    def init(self, polygons: list[Polygon]):
        points = [p.points.numpy() for p in polygons]
        counts = [len(p) for p in points]
        self.vertices = wp.array(
            np.concatenate(points) if points else np.zeros((0, 2)),
            dtype=wp.vec2,
        )
        self.offsets = wp.array(
            np.concatenate([[0], np.cumsum(counts)]), dtype=int
        )

    def __len__(self):
        return len(self.offsets) - 1


@wp.struct
class PolygonQuery:
    result: bool
    sign: float
    point: wp.vec2
    edge: int


@wp.func
def polygon_contains(
        vertices: wp.array[wp.vec2], start: int, end: int, p: wp.vec2
) -> bool:
    n = end - start
    crossings = wp.int32(0)

    for e in range(n):
        vi = vertices[start + e]
        vj = vertices[start + (e + n - 1) % n]

        above_i = vi[1] > p[1]
        above_j = vj[1] > p[1]
        crosses = wp.bool(False)
        if above_i != above_j:
            edge_x = (vj[0] - vi[0]) * (p[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]
            crosses = p[0] < edge_x
        crossings = crossings + int(crosses)

    return (crossings % 2) == 1


@wp.func
def polygon_query_point_sign_normal(
        obstacles: Obstacles, b: int, point: wp.vec2, max_dist: float
) -> PolygonQuery:
    start = obstacles.offsets[b]
    end = obstacles.offsets[b + 1]
    n = end - start

    query = PolygonQuery()
    query.result = False
    query.sign = 1.0
    query.point = point
    query.edge = 0

    best_dist_sq = max_dist * max_dist
    for e in range(n):
        v0 = obstacles.vertices[start + e]
        v1 = obstacles.vertices[start + (e + 1) % n]
        edge = v1 - v0
        t = wp.clamp(
            wp.dot(point - v0, edge) / wp.max(wp.dot(edge, edge), 1e-12),
            0.0,
            1.0,
        )
        closest = v0 + t * edge
        dist_sq = wp.length_sq(point - closest)
        if dist_sq <= best_dist_sq:
            best_dist_sq = dist_sq
            query.result = True
            query.point = closest
            query.edge = e

    if query.result and polygon_contains(obstacles.vertices, start, end, point):
        query.sign = -1.0

    return query


@wp.func
def polygon_edge_normal(obstacles: Obstacles, b: int, edge: int) -> wp.vec2:
    start = obstacles.offsets[b]
    n = obstacles.offsets[b + 1] - start
    v0 = obstacles.vertices[start + edge]
    v1 = obstacles.vertices[start + (edge + 1) % n]
    d = v1 - v0
    # outward normal of a counter-clockwise polygon
    return wp.normalize(wp.vec2(d[1], -d[0]))
