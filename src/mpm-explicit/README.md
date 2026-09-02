# mpm-explicit

> Work in progress!

A 3D Material Point Method (MPM) simulator implemented in Python using the Nvidia Warp library, and following the implementation of [1].

## Structure

- `src/mpm-explicit/constants.py`: Contains the constants used in the simulation, such as the grid size, time step, and material properties.
- `src/mpm-explicit/particle.py`: Contains the Particles Warp struct, which holds the packed particle data.
- `src/mpm-explicit/grid.py`: Contains the Grid Warp struct, which holds the packed grid data.
- `src/mpm-explicit/solver.py`: Contains the MPM solver, which implements the MPM algorithm using the Nvidia Warp library.
- `src/mpm-explicit/renderer.py`: Contains the renderer, which visualizes the simulation using [`rerun`](https://rerun.io/) library.
- `src/mpm-explicit/utils.py`: Contains utility functions and kernels for the simulation.
- `src/mpm-explicit/__main__.py`: The main entry point of the simulation, which initializes the particles and grid, and runs the simulation loop.

## Running

To run the simulation, please first check the simulation parameters in `src/mpm-explicit/constants.py` and `src/mpm-explicit/__main__.py`. Then, run the following command:

```bash[]()
uv run --package mpm-explicit python -m mpm_explicit
```

---

[1]: A. Stomakhin, C. Schroeder, L. Chai, J. Teran, and A. Selle, “A material point method for snow simulation,” ACM Transactions on Graphics, vol. 32, no. 4, pp. 1–10, Jul. 2013, doi: 10.1145/2461912.2461948.
