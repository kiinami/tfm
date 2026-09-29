# mpm13

> Work in progress!

<div align="center">
  <img src="assets/recordings/snowball_on_diamond_at_30kmph.gif" alt="Smoke Simulation Capture" width="800" />
</div>

A 3D Material Point Method (MPM) simulator implemented in Python using the Nvidia Warp library, and following the implementation of [^1].

The grid velocity update can be either explicit or semi-implicit (solved with conjugate residuals, weighted by `IMPLICIT_BETA`). The method is chosen when instantiating the solver, and can be switched at runtime:

```python
from mpm13.solver import Method, Solver

solver = Solver(grid, particles, obstacles, method=Method.IMPLICIT)
solver.method = Method.EXPLICIT
```

The whole step, including the implicit solve, is captured in a single CUDA graph; the implicit part is a `wp.capture_if` branch driven by a device flag, so switching methods does not require recapturing.

## Structure

- `src/mpm13/constants.py`: Contains the constants used in the simulation, such as the grid size, time step, and material properties.
- `src/mpm13/particle.py`: Contains the Particles Warp struct, which holds the packed particle data.
- `src/mpm13/grid.py`: Contains the Grid Warp struct, which holds the packed grid data.
- `src/mpm13/solver.py`: Contains the MPM solver, which implements the MPM algorithm using the Nvidia Warp library.
- `src/mpm13/renderer.py`: Contains the renderer, which visualizes the simulation using [`rerun`](https://rerun.io/) library.
- `src/mpm13/utils.py`: Contains utility functions and kernels for the simulation.
- `src/mpm13/__main__.py`: The main entry point of the simulation, which initializes the particles and grid, and runs the simulation loop.

## Running

To run the simulation, please first check the simulation parameters in `src/mpm13/constants.py` and `src/mpm13/__main__.py`. Then, run the following command:

```bash
uv run --package mpm13 python -m mpm13
```

---

[^1]: A. Stomakhin, C. Schroeder, L. Chai, J. Teran, and A. Selle, “A material point method for snow simulation,” ACM Transactions on Graphics, vol. 32, no. 4, pp. 1–10, Jul. 2013, doi: 10.1145/2461912.2461948.
