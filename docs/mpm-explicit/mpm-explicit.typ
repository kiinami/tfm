#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

= Common symbols

- $x$ #sym.arrow current state of the material
- $X$ #sym.arrow initial state of the material
- $rho$ #sym.arrow density
- $(partial A) / (partial B)$ #sym.arrow partial derivative of $A$ with respect to $B$
- $D / (D t)$ #sym.arrow material derivative, that is, the rate of change following a moving particle of the material (a "material point")
- $v(t)$ / $v$ #sym.arrow velocity of a material point ($3 times 1$)
- $a(t) = (D v(t)) / (D t)$ / $a$ #sym.arrow acceleration of a material point ($3 times 1$)
- $sigma$ #sym.arrow Cauchy stress ($3 times 3$, symmetric)
- $Psi$ #sym.arrow the elasto-plastic energy density: stored elastic enerygy per unit rest volume, depends on the material
- $F_E$ is the the elastic part of $F$, because $F$ can be decomposed in $F_E$ (elastic, non permanent deformation) and $F_P$ (plastic, permanent deformation) such that $F = F_E F_P$. $F_E^T$ is the transposed of $F_E$. We can also define $J_E$ and $J_P$ as their respective determinants.
- $P = (partial Psi) / (partial F_E)$ ($3 times 3$) #sym.arrow the first Piola-Kirchoff stress, it measures the force per unit rest area

= Fundamentals

- MPM uses the same hybrid method as PIC/FLIP, where the authoritative space is the Lagrangian space and the Eulerian space is used temporarily on each iteration, but it uses the same general ideas for the material continuum as FEM. That means that it has deformation, and a deformation map $x = phi(X)$
- the deformation gradient is the spatial derivative of the deformation map is $ F = (partial phi) / (partial X) \ F arrow 3 times 3 text("matrix") $
  -  $F_[r, c] = (partial phi_r) / (partial X_c)$ #sym.arrow how much the r-th coord. of the current position changes when you nudge the c-th coord. of the rest position
  - The determinant of $F$ is the local volume ratio and gets its own letter: $ J = det(F) $
    - $J = 1$ means the volume is preserved locally, $J < 1$ means compression, $J > 1$ means expansions, and $J <= 0$ means zero volume or inversion (turned inside-out)

#divider()

- The motion of any continuum obeys two conservation laws:
  - Conservation of mass: $ (D rho) / (D t) = 0 $ It says that the density carried by a particle does not change, because the particles have fixed mass. The density may change in space because the particles can cluster and spread around.
  - Conservation of momentum: $ rho dot a = nabla dot sigma + rho dot g $ It says that density times acceleration is equal to internal elastic forces per volume plus gravity per volume. It is the same as the Navier-Stokes momentum equation but with pressure/viscosity stress replaced with elastic stress.

#divider()

- $sigma$ is calculated as: $ sigma = 1/J dot P dot F_E^T $
  - The combination $1/J dot P$ is the conversion from the rest-area-based P-K stress to the current-area-based Cauchy stress
  - Thus, there are two stresses: $P$ and $sigma$. $P$ is what you naturally get by differentiating the energy (which lives in rest space), while $sigma$ is what the momentum equation needs (and lives in world space)

- The plasticity model for this method has, on one side, the $F_P$ described above, and on the other, two more components:
  - A yield criterion, a rule that determines how much elastic deformation the material tolerates before it starts flowing plastically
  - A hardening rule, which describes how the material's stiffness changes as it deforms plastically.

#divider()

- Polar decomposition: $ A = R S $ where $A$ is the matrix being decomposed (in our case, it would be $F$), $R$ is a rotation matrix, the "rotation part" of the deformation, and $S$ is the "pure stretch part" of the deformation and is symmetric positive-definite.
  - Meaning is that any local deformation is a pure stretch followed by a rotation

- Singular value decomposition: $ A = U Sigma V^T $ where $A$ is the matrix being decomposed, $U$ is an orthogonal $3 times 3$ rotation (maybe with reflection), $Sigma$ is a diagonal $3 times 3$ matrix whose diagonal values are the pure stretch factors along the three principal axes, and $V^T$ is the transpose of another orthogonal matrix $V$ that also describes a rotation.
  - Meaning: rotate into a special frame ($V^T$), stretch along the axes ($Sigma$), then rotate out.
  - Gives polar descompositon for free: $ R = U V^T , S = V Sigma V^T $

#divider()

- Material stiffness is specified by two parameters that are then consumed in the energy function as two derived numbers
  - $E$ is Young's modulus (pascals) and measures resistance to stretching.
  - $nu$ is Poisson's ratio and measures how much the material bulges sideways when compressed
  - Derived paremeters (Lamé parameters): $ mu = E / (2 (1 + nu)) \ lambda = (E nu) / ((1 + nu)(1 - 2 nu)) $

#divider()

- The model specifically for snow is defined by controllable functions:
  #let FE = $bold(F)_E$
  #let FP = $bold(F)_P$
  #let shear = $mu(#FP)$
  #let lame = $lambda(#FP)$
  - The hardening, defined by the Lamé parameters: $ #shear = mu_0 e^(xi (1 - J_p)) \ #lame = lambda_0 e^(xi (1 - J_p)) $ where $xi$ is the hardening coefficient and $mu_0$ and $lambda_0$ are the initial values, computed from inputs $E$ and $nu$ as described above
  - The energy density function, defined by $ Psi(#FE, #FP) = #shear ||#FE - bold(R)_E||_F^2 + #lame/2 (bold(J)_E - 1)^2 \  $
  - The yield criterion, that says that values of #FE must be restricted to the interval $[1 - theta_c, 1 + theta_s]$, where $theta_c$ and $theta_s$ are inputs
  - The split of $F_p$ into #FE and #FP with the following process:

#divider()




= Material Point Method

== Interpolation

- Interpolation between particles and grid is done with cubic b-spline weights: $ N_bold(i)^h (bold(x)_p) = N(1/h (x_p - i h)) dot N(1/h (y_p - j h)) dot N(1/h (z_p - k h)) $ where:
  - $h$ is the grid step size
  - $bold(i)$ (note the bold) $ = (i, j, k)$ is the integer index triplet of a grid node such that its position is $(i dot h, j dot h, k dot h)$
  - $bold(x)_p$ (note the bold) $ = (x_p, y_p, z_p)$ is the world position of the particle
  - $N(x)$ is $ N(x) = cases(
      1/2 abs(x)^3 - x^2 + 2/3 & "if" 0 <= abs(x) < 1,
      -1/6 abs(x)^3 + x^2 -2 abs(x) + 4/3 & "if" 1 <= abs(x) < 2,
      0 & "otherwise"
    ) $ with derivative $ N'(x) = cases(
      3/2 x abs(x) - 2x & "if" 0 <= abs(x) < 1,
      -1/2 x abs(x) + 2 x - 2 op("sign")(x) & "if" 1 <= abs(x) < 2,
      0 & "otherwise"
    ) $
- We define $ w_(bold(i) p) = N_bold(i)^h (bold(x)_p) $ as the scalar interpolation weight between grid node $bold(i)$ and particle $p$, and $ nabla w_bold(i)^p = nabla N_bold(i)^h (bold(x)_p) $ as the gradient of that weight with respect to the particle position $bold(x)_p$, which computed axis-by-axis with the product rule: $ u = (x_p - i dot h)/h, v = (y_p - j dot h)/h, w = (z_p - k dot h)/h \ nabla w_(bold(i)p) = 1/h mat(N'(u), N(v), N(w); N(u), N'(v), N(w); N(u), N(v), N'(w)) $

== Particle and grid states

- Each particle stores:
  - Varying properties:
    - Position ($bold(x)_p$, $3 times 1$)
    - Velocity ($bold(v)_p$, $3 times 1$)
    - Elastic deformation gradient ($F_(E p)$, $3 times 3$, $F_(E p)^0 = I$)
    - Plastic deformation gradient ($F_(P p)$, $3 times 3$, $F_(P p)^0 = I$)
  - Fixed properties:
    - Mass ($m_p$, scalar)
    - Initial volume ($V^0_p$, scalar)
    - (optional) Per-particle material parameters ($theta_c, theta_s, xi, mu_0, lambda_0$)

- The grid stores nothing between iterations, but inside an iteration each node $bold(i)$ has:
  - Mass ($m_i$, scalar)
  - Velocity ($bold(v)_i$, $3 times 1$)
  - Force ($bold(f)_i$, $3 times 1$)
  - Other solver temporary values

== Steps

We are in Lagrangian space

1. Rasterize particles into grid (P2G)
  - Transfer mass according to $ m_bold(i)^n = sum_p m_p w_(bold(i)p)^n $ which technically loops over every cell but in practice we only need to use the particles within the node's 2-cell support contribute
  - Transfer velocity according to $ bold(v)_bold(i)^n = (sum_p bold(v_p^n) m_p w_(bold(i) p)^n) / m_bold(i)^n $ which must include the mass to preserve momentum
  - Nodes with 0 or \~0 mass must be skipped or zeroed to avoid dividing by 0 or \~0. Create a list/bitmask of active nodes and do all subsequent work only with those.

We are now in Eulerian space

2. (First timestep only) estimate per-particle rest volumes $V_p^0$ from the rasterized density
  - First we need to estimate density at the particle by pushing grid mass back down with $ rho_p^0 = sum_i (m_bold(i)^0 w_(bold(i) p)^0) / h^3 $ then calculate initial volume with $ V_p^0 = m_p / rho_p^0 $

3. Compute grid forces from particle stresses
  - Elastic forces are evaluated at grid nodes from the stresses of nearby particles with $ bold(f_i) = - sum_p V_p^n sigma_p nabla w_(bold(i)p)^n = - sum_p V_p^0 P_p (F_(E p)^n)^T nabla w_(bold(i)p)^n  \ P_p = 2 mu(F_(P p)) (F_(E p) - R_(E p)) + lambda(F_(P p)) (J_(E p) - 1) J_(E p) F_(E p)^(-T) $

4. Update grid velocities explicitly with grid forces
  - With the formula $ bold(v_i)^star = bold(v_i)^n + Delta t m_bold(i)^(-1) bold(f_i)^n $

5. Resolve collisions of grid-node velocities against scripted bodies
  - Calculate the function $phi(bold(x))$ for each node for each scripted object, it being the signed distance from point $bold(x)$ to the object's surface
  - Given the velocity $bold(v)$ to be collided at a point where $phi <= 0$:
    1. Move the object's reference frame $ bold(v)_"rel" = bold(v) - bold(v)_"co" $ where $bold(v)_"co"$ is the collision object's velocity at this point
    2. Split into normal and tangential parts $ v_n = bold(v)_"rel" dot bold(n), space.quad bold(v)_t = bold(v)_"rel" - v_n bold(n) $ where $bold(n)$ is the outward unit normal ($= nabla phi$)
    3. If $v_n >= 0$ the bodies are separating, so apply no response, leaving $bold(v)$ untouched
    4. Otherwise apply Coulomb friction: $ bold(v')_"rel" = cases(0 & "if" ||bold(v)_t|| <= mu v_n, bold(v)_t + mu v_n bold(v)_t / (||bold(v)_t ||) & "otherwise" ) $
  - For surfaces where snow should stick, set $v'_"rel" = 0$ unconditionally if the surface is marked as sticky

6. Implicitly solve for the end-of-step grid velocities (or skip for explicit)
  - Solve the system $ sum_j (I delta_bold(i j) + beta delta t^2 m_bold(i)^(-1) (partial^2 Phi^n)/(partial hat(bold(x))_bold(j) partial hat(bold(x))_bold(j))) bold(v_j)^(n + 1) = bold(v_i)^star \ delta_bold(i j) = cases(1 & "if" i = j, 0 & "otherwise") \ beta = cases(0 & "for explicit integration", 1/2 & "for trapezoidal integration", 1 & "for backwards Euler") $ // TODO: desgranar con cosas en Part IV Section 4.4
7. Update deformation gradients on particles from the new grid velocity field, applying the pastic yield (SVD clamp)
  - The new grid velocity field is defined for each particle as $ nabla bold(v)_p^(n+1) = sum_i bold(v_i)^(n+1) (nabla w_(bold(i)p)^(n))^T $ where $sum_i$ iterates over the particle's support, $bold(v_i)^(n+1)$ is the new velocity calculated in step 4 if explicit or in step 6 for implicit, and $(nabla w^n_(bold(i)p))^T$ is the transpose (row 3-vector) of the weight gradient between node $bold(i)$ and particle $p$
  - Then we calculate the total deformation gradient with $ bold(F)^(n+1)_p = (I + Delta t nabla bold(v)_p^(n+1)) bold(F)_p^n $
  - Then we separate $F_p^(n+1)$ into $F_(P p)^(n+1)$ and $F_(E p)^(n+1)$ with the following process
    1. Tentatively define $ bold(hat(F))_(E p)^(n+1) = (I + Delta t nabla bold(v)_p^(n+1)) bold(F)_(E p)^n space.quad "and" space.quad bold(hat(F))_(P p)^(n+1) = bold(F)_(P p)^n $ such that the total gradient is defined by $ bold(F)_p^(n+1) = (I + Delta t nabla bold(v)_p^(n+1)) bold(F)_(E p)^n bold(F)_(P p)^n = bold(hat(F))_(E p)^(n+1) bold(hat(F))_(P p)^(n+1) $
    2. Enforce yield by clamping, first by computing the SVD of the tentative elastic part with $ bold(hat(F))_(E p)^(n+1) = U_p hat(Sigma)_p V_p^T $ then by clamping the singular values with $ Sigma_p = op("clamp")(hat(Sigma)_p, [1 - theta_c, 1 + theta_s ]) $
    3. Reassemble the final elastic and plastic parts with $ F_(E p)^(n+1) = U_p Sigma_p V_p^T space.quad "and" space.quad F_(P p)^(n+1) = V_p Sigma_p^(-1) U_p^T F_p^(n+1) $
8. Update particle velocities with PIC/FLIP blend (G2P)
  - $ bold(v)_p^(n+1) = (1 - alpha) bold(v)_("PIC"p)^(n+1) + alpha bold(v)_("FLIP"p)^(n+1) \ bold(v)_("PIC"p)^(n+1) = sum_i bold(v)_bold(i)^(n+1) w_(bold(i)p)^n \ bold(v)_("FLIP"p)^(n+1) = bold(v)_p^n + sum_i (bold(v)_bold(i)^(n+1) - bold(v)_bold(i)^n) w_(bold(i)p)^n $

We are back to Lagrangian space

9. Resolve collisions of particle velocities against scripted bodies
  - // Part V, section 5.1
10. Advect (move) particles with their new velocities with $ bold(x)_p^(n+1) = bold(x)_p^n + Delta t bold(v)_p^(n+1) $

11. Clear the grid and start the next step

