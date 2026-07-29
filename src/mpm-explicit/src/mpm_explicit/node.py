import warp as wp

@wp.struct
class Node:
    """
    The definition of a node of the Eulerian grid of the simulation

    Attributes:
        position (wp.vec3, constant): The node's position
        mass (float): The node's stored mass
        velocity (wp.vec3): The node's current velocity
        force (wp.vec3): The node's computed force
    """
    position: wp.vec3

    mass: float
    velocity: wp.vec3
    force: wp.vec3
