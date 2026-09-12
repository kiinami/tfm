#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: "The Material Point Method for Simulating Continuum Materials: A Summary",
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax")
)

#figure(
  table(
    columns: (auto, auto, auto),
    table.header("Variable", "Type", "Meaning"),
    $d$, "scalar", [Number of dimensions (2 or 3)],
    $Chi$, "theory", [Material (undeformed) space],
    $chi$, "theory", [World (deformed) space],
    $Phi$, "theory", [Deformation map / Flow map],
    $F$, "theory", [Deformation gradient],
    $J$, "theory", [Determinant of $F$]
  )
)

= Kinematics

- Particles in MPM are not particles, they are a discretization of the continuum material

== Continuum motion

- Kinematics is the study of motion occured in continuum materials, the main focus being the deformation or change in shape
- The deformation in continuum mechanics is represented with the material/undeformed space $Chi$, the world/deformed space $chi$ and a deformation map $Phi(Chi, chi)$. we can treat $Chi$ as the "initial position" and $chi$ as the "current position", such that at time $t = 0$, $Chi = chi$
- A more detailed definition: we consider the motion of material to be determined by a mapping $Phi(dot, t) : Omega^0 arrow Omega^t$ for $Omega^0, Omega^t subset RR^d$. Points in the set $Omega^0$ are material points and are denoted as $Chi$. Points in the set $Omega^t$ represent the location of material points at time $t$, and are refered to as $chi$. Thus, $Phi$ describes the motion of each material point $Chi in Omega^0$ over time: $ chi = chi(Chi, t) = Phi(Chi, t) $
- For example, the velocity of a given material point $Chi$ at time $t$ is $ V(Chi, t) = (partial Phi) / (partial t) (Chi, t) $ and the acceleration is $ A(Chi, t) = (partial^2 Phi) / (partial t^2) (Chi, t) = (partial V) / (partial t) (Chi, t) $

== Deformation

- The Jacobian of the deformation map $Phi$ is very useful. It is denoted as $F$: $ F(Chi, t) = (partial Phi) / (partial Chi) (Chi, t) = (partial chi) / (partial Chi) (Chi, t) $. It is a $d times d$ matrix.
- It can also be though of as $F(dot, t) : Omega^0 arrow RR^(d times d)$.
- In other words, for every material point $Chi$, $F(Chi, t)$ is the $RR^(d times d)$ matrix describing the deformation Jacobian of the material at time $t$. We can also use the index notation: $ F_(i j) = (partial Phi_i) / (partial Chi_j) = (partial chi_i) / (partial Chi_j), space.quad i, j = 1, ..., d $
- $F$ for transformations like velocity or traslations is equal to the identity matrix. For a rotation $R$, $F = R$. Intuitively, $F$ measures "local" rotation and as such does not change with rigid transformations.
- $J = det(F)$. Measures ration of infinitesimal volume change in the material when in $Omega^t$ to the original $Omega^0$. For rigid motions, $J = 1$. $J > 1$ means volume increase, and $J < 1$ means volume decrease. $J = 0$ means the material looses all volume, something impossible. $J < 0$ means the material has been inverted.

== Push Forward and Pull Back

- So far we have assumed a "Lagrangian" view, in which quantities are in terms of $(Chi, t)$ and the mapping $Phi$ is assumed to be bijective.
- We also assumed that it is smooth, thus sets $Omega^0$ and $Omega^t$ are homeomorphic/diffeomorphic under $Phi$. This means that no two different particles of material ever occupy the same space at the same time. This means that $forall chi in Omega^t, exists!Chi in Omega^0 "such that" Phi(Chi, t) = chi$
