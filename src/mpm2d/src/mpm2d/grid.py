from functools import cached_property

import warp as wp


@wp.struct
class Grid:
    min_coord: wp.vec2
    max_coord: wp.vec2
    dimensions: wp.vec2ui
    cell_size: wp.vec2

    masses: wp.array2d[float]
    velocities: wp.array2d[wp.vec2]
    new_velocities: wp.array2d[wp.vec2]
    forces: wp.array2d[wp.vec2]

    @cached_property
    def flat_dimensions(self) -> int:
        return self.dimensions[0] * self.dimensions[1]

    def init(self, min_coord: wp.vec2, max_coord: wp.vec2, dimensions: wp.vec2ui):
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.dimensions = dimensions
        self.cell_size = wp.vec2(
            (max_coord[0] - min_coord[0]) / float(dimensions[0]),
            (max_coord[1] - min_coord[1]) / float(dimensions[1]),
        )

        self.masses = wp.zeros(shape=self.dimensions, dtype=float)
        self.velocities = wp.zeros(shape=self.dimensions, dtype=wp.vec2)
        self.new_velocities = wp.zeros(shape=self.dimensions, dtype=wp.vec2)
        self.forces = wp.zeros(shape=self.dimensions, dtype=wp.vec2)

    def clear(self):
        self.masses.zero_()
        self.velocities.zero_()
        self.new_velocities.zero_()
        self.forces.zero_()


@wp.func
def grid_index_to_coord(grid: Grid, i: wp.int32, j: wp.int32) -> wp.vec2:
    return wp.vec2(
        grid.min_coord[0] + float(i) * grid.cell_size[0],
        grid.min_coord[1] + float(j) * grid.cell_size[1],
    )
