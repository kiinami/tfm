import warp as wp

from mpm_explicit.solver import Solver

wp.config.print_launches = True

solver = Solver(min_coord=wp.vec3(0, 0, 0), max_coord=wp.vec3(100, 100, 100),
                dimensions=wp.vec3ui(wp.uint32(100), wp.uint32(100), wp.uint32(100)),
                particles=[])
