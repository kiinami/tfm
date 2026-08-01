import numpy as np
import rerun as rr


def init(min_coord: list[float], max_coord: list[float]):
    rr.init("mpm", spawn=True)
    rr.log("world/box", rr.Boxes3D(centers=[
        [(min_coord[0] + max_coord[0]) / 2, (min_coord[1] + max_coord[1]) / 2, (min_coord[2] + max_coord[2]) / 2]],
        half_sizes=[[(max_coord[0] - min_coord[0]) / 2, (max_coord[1] - min_coord[1]) / 2,
                     (max_coord[2] - min_coord[2]) / 2]]), static=True)


def render(step: int, positions: np.ndarray):
    rr.set_time("step", sequence=step)
    rr.log("world/particles", rr.Points3D(positions=positions))
