from logging import critical

import warp as wp
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

import mpm_implicit.renderer as rd
from mpm_implicit.grid import Grid
from mpm_implicit.particles import Particles
from mpm_implicit.solver import Solver

DURATION = 1.0
FPS = 30
LOG_PER_STEP = False

PARTICLES_PER_CELL = 8

THROW_SPEED_X = 10.0
THROW_SPEED_Y = 2.6
BALL_RADIUS = 0.2
BALL_HEIGHT = 0.7


class SimTimeColumn(ProgressColumn):
    """Shows simulated seconds completed vs total, e.g. '0.42s / 1.00s'."""

    def render(self, task):
        total = task.total or 0.0
        return Text(f"{task.completed:.3f}s / {total:.3f}s", style="progress.download")


def _format_duration(seconds: float) -> str:
    """Formats a duration picking s/min/h so it stays readable at any scale."""
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.2f} min"
    return f"{seconds / 3600.0:.2f} h"


class RealtimeSpeedColumn(ProgressColumn):
    """Shows compute time spent per second of simulated time."""

    def render(self, task):
        elapsed = task.elapsed
        sim_time = task.completed
        if not elapsed or sim_time <= 0.0:
            return Text("-- compute/s sim", style="progress.data.speed")
        return Text(f"{_format_duration(elapsed / sim_time)} compute/s sim", style="progress.data.speed")


def main():
    print("Initializing warp and compiling kernels")
    wp.init()

    grid = Grid()
    grid.init(
        min_coord=wp.vec3(-1.0, -1.00, 0.0),
        max_coord=wp.vec3(1.0, 1.0, 1.5),
        dimensions=wp.vec3ui(wp.uint32(200), wp.uint32(200), wp.uint32(300)),
    )

    particles = Particles()
    particles.add_snowball(
        center=wp.vec3(0.0, -0.7, BALL_HEIGHT),
        radius=BALL_RADIUS,
        initial_velocity=wp.vec3(0.0, THROW_SPEED_X, THROW_SPEED_Y),
        critical_compression=1.9e-2,
    )
    particles.add_snowball(
        center=wp.vec3(0.0, 0.7, BALL_HEIGHT),
        radius=BALL_RADIUS,
        initial_velocity=wp.vec3(0.0, -THROW_SPEED_X, THROW_SPEED_Y),
        critical_compression=1.9e-2,
    )

    solver = Solver(grid, particles, [])

    rd.init(grid, [])

    frame_duration = 1.0 / FPS
    next_frame_time = 0.0

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            SimTimeColumn(),
            "•",
            RealtimeSpeedColumn(),
            "•",
            TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(description="Simulating...", total=DURATION)
        while solver.t < DURATION:
            solver.update()
            if LOG_PER_STEP or solver.t >= next_frame_time:
                rd.render(solver.t, solver.particles.positions.numpy())
                next_frame_time += frame_duration
            progress.update(task, completed=solver.t)


if __name__ == "__main__":
    main()
