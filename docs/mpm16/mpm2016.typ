#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: "The Material Point Method for Simulating Continuum Materials: A Summary",
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax")
)

#let O0 = $Omega^0$
#let Ot = $Omega^t$
#let ij = $i j$
#let dimsum = $sum^d_(i=1)$

#figure(
  table(
    columns: (auto, auto, auto),
    table.header("Variable", "Type", "Meaning"),
    $d$, "scalar", [Number of dimensions (2 or 3)],
    $Chi$, "theory", [Material (undeformed) space],
    $chi$, "theory", [World (deformed) space],
    $Phi$, "theory", [Deformation map / Flow map],
    $F$, "theory", [Deformation gradient],
    $J$, "theory", [Determinant of $F$],
    $O0$, "theory", [Set of points in $Chi$],
    $Ot$, "theory", [Set of points in $chi$]
  )
)

= Kinematics

- Particles in MPM are not particles, they are a discretization of the continuum material

== Continuum motion

- Kinematics is the study of motion occured in continuum materials, the main focus being the deformation or change in shape
- The deformation in continuum mechanics is represented with the material/undeformed space $Chi$, the world/deformed space $chi$ and a deformation map $Phi(Chi, chi)$. we can treat $Chi$ as the "initial position" and $chi$ as the "current position", such that at time $t = 0$, $Chi = chi$
- A more detailed definition: we consider the motion of material to be determined by a mapping $Phi(dot, t) : O0 arrow Ot$ for $O0, Ot subset RR^d$. Points in the set $O0$ are material points and are denoted as $Chi$. Points in the set $Ot$ represent the location of material points at time $t$, and are refered to as $chi$. Thus, $Phi$ describes the motion of each material point $Chi in O0$ over time: $ chi = chi(Chi, t) = Phi(Chi, t) $
- For example, the velocity of a given material point $Chi$ at time $t$ is $ V(Chi, t) = (partial Phi) / (partial t) (Chi, t) $ and the acceleration is $ A(Chi, t) = (partial^2 Phi) / (partial t^2) (Chi, t) = (partial V) / (partial t) (Chi, t) $

== Deformation

- The Jacobian of the deformation map $Phi$ is very useful. It is denoted as $F$: $ F(Chi, t) = (partial Phi) / (partial Chi) (Chi, t) = (partial chi) / (partial Chi) (Chi, t) $. It is a $d times d$ matrix.
- It can also be though of as $F(dot, t) : O0 arrow RR^(d times d)$.
- In other words, for every material point $Chi$, $F(Chi, t)$ is the $RR^(d times d)$ matrix describing the deformation Jacobian of the material at time $t$. We can also use the index notation: $ F_(ij) = (partial Phi_i) / (partial Chi_j) = (partial chi_i) / (partial Chi_j), space.quad i, j = 1, ..., d $
- $F$ for transformations like velocity or traslations is equal to the identity matrix. For a rotation $R$, $F = R$. Intuitively, $F$ measures "local" rotation and as such does not change with rigid transformations.
- $J = det(F)$. Measures ration of infinitesimal volume change in the material when in $Ot$ to the original $O0$. For rigid motions, $J = 1$. $J > 1$ means volume increase, and $J < 1$ means volume decrease. $J = 0$ means the material looses all volume, something impossible. $J < 0$ means the material has been inverted.

== Push Forward and Pull Back

- So far we have assumed a "Lagrangian" view, in which quantities are in terms of $(Chi, t)$ and the mapping $Phi$ is assumed to be bijective.
- We also assumed that it is smooth, thus sets $O0$ and $Ot$ are homeomorphic/diffeomorphic under $Phi$. This means that no two different particles of material ever occupy the same space at the same time. This means that $forall chi in Ot, exists!Chi in O0 "such that" Phi(Chi, t) = chi$
- This means that any function over one set can be thought of as a function over the other set by changing variables
- We can call this variable change "push forward" (when we go from $Omega_0$ to $Omega_t$) or "pull back" (from $Omega_t$ to $Omega_0$)
- The "push forward" version of a function is sometimes called the Eulerian, and it is a function of $chi$. Conversely, the "pull back" version is called the Lagrangian, and it is a function of $Chi$
- We have defined velocity and acceleration in Lagrangian: $ V(Chi, t) = (partial Phi) / (partial t) (Chi, t) $ $ A(Chi, t) = (partial^2 Phi) / (partial t^2) (Chi, t) = (partial V) / (partial t) (Chi, t) $
  - We can define the Eulerian counterparts: $ v(chi, t) = V(Phi^(-1)(chi, t), t) $ $ a(chi, t) = A(Phi^(-1)(chi, t), t) $
  - With this, we can say that the pull back formulae can also be written as: $ V(Chi, t) = v(Phi(Chi, t), t) $ $ A(Chi, t) = a(Phi(Chi, t), t) $
  - Using the chain rule, we can see that $ A(Chi, t) = partial/(partial t) V(Chi, t) \ = (partial v)/(partial t)(Phi(Chi, t), t) + (partial v)/(partial x)(Phi(Chi, t), t) (partial Phi)/(partial t)(Chi, t) $
  - We can rewrite this using indexes: $ A_i (Chi, t) = (partial)/(partial t) V_i (Chi, t) \ = (partial v_i)/(partial t) V_i (Phi(Chi, t), t) + (partial v_i)/(partial x_j)(Phi(Chi, t), t) (partial Phi_j)/(partial t)(Chi, t) $
- Combining the equations for $V(Chi, t)$ and $v(chi, t)$ we have $ v_j (chi, t) = (partial Phi_j)/(partial t) (Phi^(-1) (chi, t), t) $
- And combining the equations for $a(chi, t)$ and $A_i (Chi, t)$ we have $ a_i (chi, t) = A_i (Phi^(-1) (chi, t), t) \ = (partial v_i)/(partial t) (chi, t) + (partial v_i)/(partial x_j)(chi, t)v_j (chi, t) $
- And surprisingly we get with this to the conclusion that $ a_i (chi, t) != (partial v_i)/(partial t) (chi, t) $

== Material Derivative

- The relationship between the Eulerian $a$ and $v$ is not simply via partial differentiation with respect to time, as seen in the previous section.
- Instead, it is called the "material derivative"
- We introduce the notation $ D/(D t) v_i (chi, t) = (partial v_i)/(partial t) (chi, t) + (partial v_i)/(partial x_j)(chi, t)v_j (chi, t) $ so that $ a = D/(D t) v $
- For a general Eulerian function $f(dot, t) : Ot arrow RR$ we use the same notation to mean $ D/(D t) f (chi, t) = (partial f)/(partial t) (chi, t) + (partial f)/(partial x_j)(chi, t)v_j (chi, t) $
- $D/(D t) f(chi, t)$ is the push forward of $partial/(partial t) F$, where $F$ is the Lagrangian version of the function shuch that $F(dot, t) : O0 arrow RR$. $F$ is the pull back of $f$.
- The deformation gradient $F$ is normally thought of in Lagrangian space, but there is an useful evolution of the Eulerian of $F(dot, t) : O0 arrow RR^(d times d)$
  - Let $f(dot, t) : Ot arrow RR^(d times d)$ be the push forward of $F$, then $ D/(D t) f = (partial v)/(partial chi) f "or" D/(D t) f_(ij) = (partial v_i) / (partial chi_k) f_(k j) $ with summation implied on $j$

== Volume and Area Change

- Assume there is a tiny volume $d V$ at the material space. What is the corresponding value of $d  v$ in the world space?
- Consider $d V$ being defined over the standard basis vectors $e_1, e_2, e_3$ with $d V = d L_1 e_1 dot (d L_2 e_2 times d L_3 e_3)$. When $d L_i$ are tiny numbers, $d L_i = d L_i e_i$. Then we have $ d V = d L_1 d L_2 d L_3 $
- The corresponding deformed vectors in the world space are $ d l_i = F d L_i $
- It can be shown that $d l_1 d l_2 d l_3 = J d L_1 d L_2 d L_3$, that is, $d v = J d V$, where $J = det(F)$
- Given this proprerty, for any function $G(Chi)$ or $g(chi, t)$ it is common to use the push forward/pull back when changing variables for integrals defined over subsets of either $O0$ or $Ot$. That is $ integral_(B^t) g(chi) d chi = integral_(B^0) G(Chi) J (Chi, t) d Chi $ where $B^t$ is an arbitrary subset of $Ot$, $B^0$ is the pre-image of $B^t$ under $Phi(dot, t)$, $G$ is the pull back of $g$ and $J(Chi, t)$ is the determinant of the deformation gradient.
- Similar analysis can be done for areas

= Hyperelasticity

- The physical meaning of stress will be introduced properly next section, when deriving the governing equations. For now, we simply introduce the fact that stress is related to strain (or in our case, deformation gradient) through some "constitutive relationship".
- Stress is a field that exists in the whole domain.
- There are multiple stress definitions available.
- Discretely, stress is a small matrix at each evaluated point.
- To govern the responses of the material to deformations, we need a constitutive model that relates the state to the stress.
- For perfect hyperelastic materials the constitutive relation is defined through the potential energy, which increases with non-rigid deformation from the initial state.
- In this section we focus on elastic materials as well as an inexact but practically easy to use plastic model.

== First Piola-Kirchoff Stress (PK1)

- For traditional solids, it is preferred to express strain stress relationship using deformation gradient and first Piola-Kirchoff stress because they are more naturally expressed in the material space.
- Hyperelastic materials are those elastic solids whose PK1 $P$ can be derived from an strain energy density function $Psi(F)$ via $ P = (partial Psi)/(partial F) $ <PispartialPsidivbypartialF> or with index notation $ P_(ij) = (partial Psi)/(partial F_ij) $
- $Psi(F)$ is the elastic energy density function. It is scalar, and designed to penalize non-rigid $F$.
- Discretely, $P$ is a small matrix with the same dimension as $F$
- We can easily relate $P$ to the Cauchy stress $sigma$ via $ sigma = 1/J P F^T = 1/det(F) (partial Psi)/(partial F) F^T $
- The behaviour of a material is defined via the interaction of $Phi$ and the stress $sigma$ or $P$. For hyperelastic materials, the stress is a function of the change in shape, as expressed via the deformation gradient.
- The motion of the material is rigid if $ Phi(Chi, t) = R(t)Chi + bold(t)(t) $ where $R$ is the rotation and fits that $R^T R = I$, $det(R) = 1$ and $t : [0, infinity] arrow RR^d$, and $bold(t)$ is the traslation. All of which means that, for this case, $F = R$.
- In other words, the energy does not change (and has a minimum) if $F$ is orthogonal.
- $F^T F$ is often denoted with $C$ (right Cauchy-Green tensor).
- If a material is isotropic (meaning that response to deformation is material direction independent), then we can further simplify the enerfy by writing it as a function of the invariants of $C$: $ Psi = Psi(I_1, I_2, I_3) $ where $I_i$ are the coefficients of the characteristic polynomial of $C$: $ I_i = tr(C), \ I_2 = tr(C C), \ I_3 = det(C) = J^2 $
- In graphics it has been convenient to further write this as $ Psi(F) = hat(Psi)(Sigma(F)) $ where $F = U Sigma V^T$ (polar decomposition).

== Neo-Hookean

- Neo-Hookean is one of the most common nonlinear hyperelastic models for predicting large deformations of elastic materials
- The energy density function for this model is $ Psi(F) = \ mu/2 (tr(F^T F) - d) - mu log(J) + lambda/2 log^2 (J) $ where $d = 2 "or" 3$ denotes the problem dimension, and $mu$ and $lambda$ are from the classic Young's modulus $E$ and Poisson's ratio $nu$: $ mu = E/(2(1 + nu)), lambda = (E nu)/((1 + nu)(1-2 nu)) $
- It is easy to see that when $F$ is a rotation, $Psi(F) = 0$. For a non-inverted ($J > 0$) $F$, $Psi(F) >= 0$.
- The energy density function is adecuate to describe a hyperelastic solid
- Using @PispartialPsidivbypartialF, we can provide $P$ as a function of $F$ for Neo-Hookean, the result being $ P = mu(F - F^(-T)) + lambda log(J) F^(-T) $
- Note that $(partial P)/(partial F)$ will be needed for implicit integration. We will provide it later on.

== Fixed Corotated Constitutive Model

- Another simple and widely used model that is defined from the SVD
- It is called "fixed" because it is a small modification to a commonly used model called corotated linear elasticity common in the computer graphics literature.
- Assuming the polar SVD $F = U Sigma V^T$, the energy for the fixed corotated model is $ Psi(F) = hat(Psi)(Sigma(F)) = mu dimsum (sigma_i - 1)^2 + lambda/2 (J - 1)^2 \ J = product^d_(i=1) sigma_i $
- Expanding the $mu$, we have $ mu dimsum (sigma_i - 1)^2 = mu(dimsum sigma_i^2 - 2 dimsum sigma_i + d) $
- It can be shown that $ partial/(partial F) dimsum sigma^2_i = 2F "and" partial/(partial F) dimsum sigma_i = R $ where $F = R S$ is the polar decomposition of $F$ ($R$ a rotation matrix and $S$ symmetric).
- 
