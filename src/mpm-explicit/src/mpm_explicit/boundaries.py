import warp as wp


@wp.struct
class PlaneBoundary:
    point: wp.vec3
    normal: wp.vec3

    def init(self, point: wp.vec3, normal: wp.vec3):
        self.point = point
        self.normal = wp.normalize(normal)

    def hesse_normal_form(self) -> tuple[wp.vec3, float]:
        d = -wp.dot(self.normal, self.point)
        return wp.vec3(self.normal[0], self.normal[1], self.normal[2]), d

@wp.func
def signed_distance(boundary: PlaneBoundary, point: wp.vec3) -> float:
    return wp.dot(point - boundary.point, boundary.normal)
