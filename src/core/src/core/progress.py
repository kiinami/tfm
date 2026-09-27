from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class SimTimeColumn(ProgressColumn):
    def render(self, task):
        total = task.total or 0.0
        return Text(f"{task.completed:.3f}s / {total:.3f}s", style="progress.download")


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.2f} min"
    return f"{seconds / 3600.0:.2f} h"


class RealtimeSpeedColumn(ProgressColumn):
    def render(self, task):
        elapsed = task.elapsed
        sim_time = task.completed
        if not elapsed or sim_time <= 0.0:
            return Text("-- compute/s sim", style="progress.data.speed")
        return Text(f"{_format_duration(elapsed / sim_time)} compute/s sim", style="progress.data.speed")


class MPMProgress:
    def __init__(self, duration, fps=120, render_steps=False, description="Simulating..."):
        self.duration = duration
        self.fps = fps
        self.render_steps = render_steps
        self.description = description
        self.should_render = False

        self._frame_duration = 1.0 / fps
        self._next_frame_time = 0.0

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            SimTimeColumn(),
            "•",
            RealtimeSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        )
        self._task = None

    def __enter__(self):
        self._progress.__enter__()
        self._task = self._progress.add_task(description=self.description, total=self.duration)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._progress.__exit__(exc_type, exc_val, exc_tb)

    def set_t(self, t):
        if self.render_steps or t >= self._next_frame_time:
            self.should_render = True
            self._next_frame_time += self._frame_duration
        else:
            self.should_render = False
        self._progress.update(self._task, completed=t)
