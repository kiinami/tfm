# TFM

Monorepo containing the code for my Master's Thesis in Computer Graphics at URJC.

The objective of the thesis is to implement a custom user-controllable node-based material point method (MPM) simulator in Python to integrate with Houdini.

## Current projects

- **[WIP] [src/canny](src/canny/README.md)**: A Canny edge detector implemented in Python using the Nvidia Warp library.
- **[WIP] [src/mpm-explicit](src/mpm-explicit/README.md)**: A 3D Material Point Method (MPM) simulator implemented in Python using the Nvidia Warp library, and following the implementation of [1].
- **[WIP] [src/mpm16](src/mpm16/README.md)**: A 3D Material Point Method (MPM) simulator implemented in Python using the Nvidia Warp library, and following the implementation of [2].
- **[docs/](docs/README.md)**: Documentation for the thesis, including notes and references.

---

[1]: A. Stomakhin, C. Schroeder, L. Chai, J. Teran, and A. Selle, “A material point method for snow simulation,” ACM Transactions on Graphics, vol. 32, no. 4, pp. 1–10, Jul. 2013, doi: 10.1145/2461912.2461948.

[2]: C. Jiang, C. Schroeder, J. Teran, A. Stomakhin, and A. Selle, “The material point method for simulating continuum materials,” in ACM SIGGRAPH 2016 Courses, Anaheim, California: Association for Computing Machinery, 2016. doi: 10.1145/2897826.2927348.
