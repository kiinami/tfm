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

import mpm_implicit.renderer as rd
from mpm_implicit.grid import Grid
from mpm_implicit.particles import Particles
from mpm_implicit.solver import Solver
from mpm_implicit.utils import import_mesh

DURATION = 5.0
FPS = 30
LOG_PER_STEP = False

PARTICLES_PER_CELL = 8


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
        min_coord=wp.vec3(-2.0, -2.0, -0.06),
        max_coord=wp.vec3(2.0, 2.0, 4.0),
        dimensions=wp.vec3ui(wp.uint32(200), wp.uint32(200), wp.uint32(203)),
    )

    cell_volume = grid.cell_size[0] * grid.cell_size[1] * grid.cell_size[2]
    derived_density = PARTICLES_PER_CELL / cell_volume

    particles = Particles()
    particles.sample_mesh(
        mesh=import_mesh("assets/models/snow_rectangle.obj"),
        particle_density=derived_density
    )

    obstacles = [
        import_mesh("assets/models/floor.obj"),
        import_mesh("assets/models/diamond.obj"),
    ]

    solver = Solver(grid, particles, obstacles)

    rd.init(grid, obstacles)

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
