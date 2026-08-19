from functools import cached_property

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

    positions: wp.array3d[wp.vec3]

    masses: wp.array3d[float]
    velocities: wp.array3d[wp.vec3]
    new_velocities: wp.array3d[wp.vec3]
    forces: wp.array3d[wp.vec3]

    is_active: wp.array[wp.int32]
    offsets: wp.array[wp.int32]
    active_nodes: wp.array[wp.int32]

    @cached_property
    def flat_dimensions(self) -> int:
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def init(self, min_coord: wp.vec3, max_coord: wp.vec3, dimensions: wp.vec3ui):
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.dimensions = dimensions
        self.cell_size = wp.vec3(
            (max_coord[0] - min_coord[0]) / float(dimensions[0]),
            (max_coord[1] - min_coord[1]) / float(dimensions[1]),
            (max_coord[2] - min_coord[2]) / float(dimensions[2]),
        )

        self.positions = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")
        wp.launch(
            kernel=k_grid_calculate_positions,
            dim=self.dimensions,
            inputs=[self],
        )

        self.masses = wp.zeros(shape=self.dimensions, dtype=float, device="cuda")
        self.velocities = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")
        self.new_velocities = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")
        self.forces = wp.zeros(shape=self.dimensions, dtype=wp.vec3, device="cuda")

        self.is_active = wp.empty(shape=[self.flat_dimensions], dtype=wp.int32, device="cuda")
        self.offsets = wp.empty(shape=[self.flat_dimensions], dtype=wp.int32, device="cuda")
        self.active_nodes = wp.empty(shape=[self.flat_dimensions], dtype=wp.int32, device="cuda")

    def center(self) -> wp.vec3:
        return (self.min_coord + self.max_coord) * 0.5

    def half_dimensions(self) -> wp.vec3:
        return wp.vec3(
            float(self.dimensions[0]) * 0.5,
            float(self.dimensions[1]) * 0.5,
            float(self.dimensions[2]) * 0.5,
        )

    @cached_property
    def active_node_count(self):
        n = int(self.flat_dimensions)

        last_offset = int(self.offsets[n - 1: n].numpy()[0])
        last_active = int(self.is_active[n - 1: n].numpy()[0])

        return last_offset + last_active

    def clear(self):
        self.masses.zero_()
        self.is_active.zero_()
        self.offsets.zero_()
        self.active_nodes.zero_()
        self.velocities.zero_()
        self.new_velocities.zero_()
        self.forces.zero_()
        self.__dict__.pop("active_node_count", None)


@wp.kernel
def k_grid_calculate_positions(grid: Grid):
    i, j, k = wp.tid()

    grid.positions[i, j, k] = wp.vec3(
        grid.min_coord[0] + float(i) * grid.cell_size[0],
        grid.min_coord[1] + float(j) * grid.cell_size[1],
        grid.min_coord[2] + float(k) * grid.cell_size[2],
    )


@wp.func
def flatten(dimensions: wp.vec3ui, i: int, j: int, k: int) -> int:
    dim_y = int(dimensions[1])
    dim_z = int(dimensions[2])

    return (i * dim_y + j) * dim_z + k


@wp.func
def unflatten(dimensions: wp.vec3ui, idx: int) -> wp.vec3ui:
    dim_y = int(dimensions[1])
    dim_z = int(dimensions[2])

    k = idx % dim_z
    temp = idx // dim_z

    j = temp % dim_y
    i = temp // dim_y

    return wp.vec3ui(wp.uint32(i), wp.uint32(j), wp.uint32(k))
