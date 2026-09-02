import numpy as np
import trimesh
import warp as wp
from rich.progress import Progress, MofNCompleteColumn, TextColumn, ProgressColumn, BarColumn, \
    TaskProgressColumn, TimeRemainingColumn
from rich.text import Text

import mpm_explicit.renderer as rd
from mpm_explicit.grid import Grid
from mpm_explicit.particles import Particles
from mpm_explicit.solver import Solver

DURATION = 5.0
FPS = 30
DT = 1e-5

PARTICLES_PER_CELL = 64


def main():
    print("Initializing warp and compiling kernels")
    wp.init()

    grid = Grid()
    grid.init(
        min_coord=wp.vec3(-2.0, -2.0, -0.05),
        max_coord=wp.vec3(2.0, 2.0, 3.0),
        dimensions=wp.vec3ui(wp.uint32(200), wp.uint32(200), wp.uint32(305))
    )

    cell_volume = (1.10 / 110) ** 3
    derived_density = PARTICLES_PER_CELL / cell_volume

    particles = Particles()
    particles.sample_packed_snowball(
        center=wp.vec3(0.0, 0.0, 2.5),
        radius=0.2,
        particle_density=derived_density,
    )
    particles.velocities.fill_(wp.vec3(0.0, 0.0, -9.81 * 2))

    obstacles = []
    mesh_files = [
        "assets/models/floor.obj",
        "assets/models/diamond.obj",
    ]

    for f in mesh_files:
        tm = trimesh.load_mesh(f)
        rotation = trimesh.transformations.rotation_matrix(
            np.radians(90),
            [1, 0, 0]
        )
        tm.apply_transform(rotation)
        points = np.asarray(tm.vertices, dtype=np.float32)
        indices = np.asarray(tm.faces, dtype=np.int32)
        mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device="cuda"),
            indices=wp.array(indices.reshape(-1), dtype=wp.int32, device="cuda"),
        )
        obstacles.append(mesh)

    solver = Solver(grid, particles, obstacles, DT)

    rd.init(grid, obstacles)
    rd.render(solver.t, solver.particles.positions.numpy())

    total_steps = int(round(DURATION / DT))
    frame_duration = 1.0 / FPS
    next_frame_time = 0.0
    total_frames = int(round(DURATION * FPS))

    class TimePerStepColumn(ProgressColumn):
        """Calculates and displays the average time taken per step."""

        def __init__(self, moving_average=True):
            super().__init__()
            # If True, calculates the recent average using 1 / speed.
            # If False, calculates the cumulative average since the start.
            self.moving_average = moving_average

        def render(self, task):
            if self.moving_average:
                speed = task.speed
                if not speed or speed <= 0:
                    return Text("? s/step", style="dim")
                time_per_step = 1.0 / speed
            else:
                if task.completed == 0 or task.elapsed is None:
                    return Text("? s/step", style="dim")
                time_per_step = task.elapsed / task.completed

            if time_per_step < 1.0:
                return Text(f"{time_per_step * 1000:.1f} ms/step", style="cyan")
            else:
                return Text(f"{time_per_step:.2f} s/step", style="cyan")

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            "•",
            MofNCompleteColumn(),
            "•",
            TimePerStepColumn(moving_average=False),
            "•",
            TimeRemainingColumn(),
    ) as progress:
        task_steps = progress.add_task(description="Steps", total=total_steps)
        task_frames = progress.add_task(description="Frames", total=total_frames)
        for _ in range(total_steps):
            solver.update()
            if solver.t >= next_frame_time:
                rd.render(solver.t, solver.particles.positions.numpy())
                next_frame_time += frame_duration
                progress.update(task_frames, advance=1)

            progress.update(task_steps, advance=1)


if __name__ == "__main__":
    main()
