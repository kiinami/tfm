from functools import cached_property

import warp as wp


@wp.struct
class Grid:
    """
    A eulerian grid calculated over a set of particles
    """
    min_coord: wp.vec2
    max_coord: wp.vec2
    dimensions: wp.vec2ui
    cell_size: wp.vec2

    masses: wp.array2d[float]
    velocities: wp.array2d[wp.vec2]
    new_velocities: wp.array2d[wp.vec2]
    forces: wp.array2d[wp.vec2]

    is_active: wp.array2d[bool]

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
        self.is_active = wp.zeros(shape=self.dimensions, dtype=bool)

    def min_size(self) -> float:
        return float(min(self.cell_size[0], self.cell_size[1]))

    def center(self) -> wp.vec2:
        return (self.min_coord + self.max_coord) * 0.5

    def half_dimensions(self) -> wp.vec2:
        return wp.vec2(
            float(self.dimensions[0]) * 0.5,
            float(self.dimensions[1]) * 0.5,
        )

    def clear(self):
        self.masses.zero_()
        self.is_active.zero_()
        self.velocities.zero_()
        self.new_velocities.zero_()
        self.forces.zero_()


@wp.func
def grid_index_from_flat(grid: Grid, idx: wp.int32) -> tuple[wp.int32, wp.int32]:
    dim_y = int(grid.dimensions[1])
    i = idx // dim_y
    j = idx % dim_y
    return (i, j)


@wp.func
def grid_index_to_flat(grid: Grid, i: wp.int32, j: wp.int32) -> wp.int32:
    return i * int(grid.dimensions[1]) + j


@wp.func
def grid_index_to_coord(grid: Grid, i: wp.int32, j: wp.int32) -> wp.vec2:
    return wp.vec2(
        grid.min_coord[0] + float(i) * grid.cell_size[0],
        grid.min_coord[1] + float(j) * grid.cell_size[1],
    )
