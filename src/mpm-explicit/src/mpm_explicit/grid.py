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

    def init(self, min_coord: wp.vec3, max_coord: wp.vec3, dimensions: wp.vec3ui):
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.dimensions = dimensions
        self.cell_size = wp.vec3(
            (max_coord[0] - min_coord[0]) / float(dimensions[0]),
            (max_coord[1] - min_coord[1]) / float(dimensions[1]),
            (max_coord[2] - min_coord[2]) / float(dimensions[2]),
        )
        self.active = wp.zeros(shape=self.dimensions, dtype=bool, device="cuda")
        self.positions = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")
        self.masses = wp.zeros(shape=self.dimensions, dtype=float, device="cuda")
        self.velocities = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")
        self.forces = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")

        wp.launch(
            kernel=k_fill_grid_constants,
            dim=self.dimensions,
            inputs=[self],
        )


@wp.kernel
def k_fill_grid_constants(grid: Grid):
    i, j, k = wp.tid()

    grid.active[i, j, k] = False
    grid.positions[i, j, k] = wp.vec3(
        grid.min_coord[0] + float(i) * grid.cell_size[0],
        grid.min_coord[1] + float(j) * grid.cell_size[1],
        grid.min_coord[2] + float(k) * grid.cell_size[2],
    )
    grid.masses[i, j, k] = 0.0
    grid.velocities[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    grid.forces[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
