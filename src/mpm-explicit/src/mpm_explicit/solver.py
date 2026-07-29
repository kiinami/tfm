import warp as wp

from mpm_explicit.grid import Grid
from mpm_explicit.particle import Particle
from mpm_explicit.utils import bspline_w, bspline_dw


@wp.kernel
def init_grid(grid: Grid):
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


@wp.struct
class Scene:
    """
    Scene data
    """
    grid: Grid
    particles: wp.array[Particle]


class Solver:
    scene: Scene = Scene()

    def __init__(self, min_coord: wp.vec3, max_coord: wp.vec3, dimensions: wp.vec3ui, particles: list[Particle]):
        self.scene.grid = Grid()
        self.scene.grid.min_coord = min_coord
        self.scene.grid.max_coord = max_coord
        self.scene.grid.dimensions = dimensions
        self.scene.grid.cell_size = wp.vec3(
            (max_coord[0] - min_coord[0]) / float(dimensions[0]),
            (max_coord[1] - min_coord[1]) / float(dimensions[1]),
            (max_coord[2] - min_coord[2]) / float(dimensions[2]),
        )
        self.scene.grid.active = wp.zeros(shape=self.scene.grid.dimensions, dtype=bool, device="cuda")
        self.scene.grid.positions = wp.zeros(shape=self.scene.grid.dimensions, dtype=wp.vec3, device="cuda")
        self.scene.grid.masses = wp.zeros(shape=self.scene.grid.dimensions, dtype=float, device="cuda")
        self.scene.grid.velocities = wp.zeros(shape=self.scene.grid.dimensions, dtype=wp.vec3, device="cuda")

        self.scene.particles = wp.array(particles, dtype=Particle)

        wp.launch(
            kernel=init_grid,
            dim=self.scene.grid.dimensions,
            inputs=[self.scene.grid],
        )

        wp.launch(
            kernel=p2g,
            dim=len(self.scene.particles),
            inputs=[self.scene.particles, self.scene.grid]
        )

        wp.launch(
            kernel=normalize_grid_velocity,
            dim=self.scene.grid.dimensions,
            inputs=[self.scene.grid]
        )


@wp.kernel
def p2g(particles: wp.array[Particle], grid: Grid):
    p = wp.tid()

    particle = particles[p]

    rel_pos = particle.position - grid.min_coord

    base_i = int(wp.floor(rel_pos[0] / grid.cell_size[0])) - 1
    base_j = int(wp.floor(rel_pos[1] / grid.cell_size[1])) - 1
    base_k = int(wp.floor(rel_pos[2] / grid.cell_size[2])) - 1

    for dk in range(4):
        for dj in range(4):
            for di in range(4):
                i = base_i + di
                j = base_j + dj
                k = base_k + dk

                if i < 0 or i >= grid.masses.shape[0] or \
                        j < 0 or j >= grid.masses.shape[1] or \
                        k < 0 or k >= grid.masses.shape[2]:
                    continue

                node_position = grid.positions[i, j, k]

                dx = (particle.position[0] - node_position[0]) / grid.cell_size[0]
                dy = (particle.position[1] - node_position[1]) / grid.cell_size[1]
                dz = (particle.position[2] - node_position[2]) / grid.cell_size[2]

                wx = bspline_w(dx)
                wy = bspline_w(dy)
                wz = bspline_w(dz)
                weight = wx * wy * wz

                wp.atomic_add(grid.masses, i, j, k, weight * particle.mass)
                wp.atomic_add(grid.velocities, i, j, k, weight * particle.mass * particle.velocity)


@wp.kernel
def normalize_grid_velocity(grid: Grid):
    i, j, k = wp.tid()

    mass = grid.masses[i, j, k]

    if mass > 0.0:
        grid.velocities[i, j, k] = grid.velocities[i, j, k] / mass
        grid.active[i, j, k] = True
    else:
        grid.velocities[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
        grid.active[i, j, k] = False
