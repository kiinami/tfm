import warp as wp


@wp.struct
class Grid:
    """
    A eulerian grid calculated over a set of particles

    Attributes
    """
    min_coord: wp.vec3
    max_coord: wp.vec3
    dimensions: wp.vec3ui
    cell_size: wp.vec3

    active: wp.array3d[bool]
    positions: wp.array3d[wp.vec3]
    masses: wp.array3d[float]
    velocities: wp.array3d[wp.vec3]

    forces: wp.array3d[wp.vec3]
