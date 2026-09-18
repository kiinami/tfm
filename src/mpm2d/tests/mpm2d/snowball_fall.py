import warp as wp
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

import mpm2d.renderer as rd
from mpm2d.grid import Grid
from mpm2d.particles import Particles
from mpm2d.solver import Solver

DURATION = 5.0
FPS = 30
DT = 1e-5
PARTICLES_PER_CELL = 16


def main():
    print("Initializing warp and compiling kernels")
    wp.init()

    grid = Grid()
    grid.init(
        min_coord=wp.vec2(0.0),
        max_coord=wp.vec2(1.0, 4.0),
        dimensions=wp.vec2ui(wp.uint32(100), wp.uint32(400)),
    )

    particles = Particles()
    particles.add_snowball(
        center=wp.vec2(0.5, 3.5),
        radius=0.1,
        segments=16,
        hardening_coef=200.0,
    )

    rd.init(grid)

    solver = Solver(grid, particles, DT)

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
        task_frames = progress.add_task(
            description="Frames", total=total_frames
        )
        for _ in range(total_steps):
            solver.update()
            if solver.t >= next_frame_time:
                rd.render(solver.t, solver.particles.positions.numpy())
                next_frame_time += frame_duration
                progress.update(task_frames, advance=1)

            progress.update(task_steps, advance=1)


if __name__ == "__main__":
    main()
