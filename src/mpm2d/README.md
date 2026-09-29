# mpm2d

> Work in progress!

A 2D Material Point Method (MPM) simulator implemented in Python using the Nvidia Warp library, and following the implementation of [^1]. It mirrors the 3D `mpm13` sub-project, only in two dimensions.

The grid velocity update can be either explicit or semi-implicit (solved with conjugate residuals, weighted by `IMPLICIT_BETA`). The method is chosen when instantiating the solver, and can be switched at runtime:

```python
from mpm2d.solver import Method, Solver

solver = Solver(grid, particles, obstacles, method=Method.IMPLICIT)
solver.method = Method.EXPLICIT
```

The whole step, including the implicit solve, is captured in a single CUDA graph; the implicit part is a `wp.capture_if` branch driven by a device flag, so switching methods does not require recapturing.

Obstacles are closed 2D polygons, built in code with `box_polygon` or `regular_polygon` from `mpm2d.utils`.

## Structure

- `src/mpm2d/constants.py`: Contains the constants used in the simulation, such as the grid size, time step, and material properties.
- `src/mpm2d/particle.py`: Contains the Particles Warp struct, which holds the packed particle data.
- `src/mpm2d/grid.py`: Contains the Grid Warp struct, which holds the packed grid data.
- `src/mpm2d/solver.py`: Contains the MPM solver, which implements the MPM algorithm using the Nvidia Warp library.
- `src/mpm2d/renderer.py`: Contains the renderer, which visualizes the simulation using [`rerun`](https://rerun.io/) library.
- `src/mpm2d/utils.py`: Contains utility functions and kernels for the simulation, including the polygon obstacles.
- `src/mpm2d/__main__.py`: The main entry point of the simulation, which initializes the particles and grid, and runs the simulation loop.

## Running

To run the simulation, please first check the simulation parameters in `src/mpm2d/constants.py` and `src/mpm2d/__main__.py`. Then, run the following command:

```bash
uv run --package mpm2d python -m mpm2d
```

---

[^1]: A. Stomakhin, C. Schroeder, L. Chai, J. Teran, and A. Selle, “A material point method for snow simulation,” ACM Transactions on Graphics, vol. 32, no. 4, pp. 1–10, Jul. 2013, doi: 10.1145/2461912.2461948.
