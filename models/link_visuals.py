"""
Shared Visual helper for DH models and Swift scene objects.

Simple cylinders for each link
"""

import numpy as np
import spatialgeometry as geometry
from spatialmath import SE3
from math import pi

def _local_vector_to_previous_frame(a, d, alpha):
    """Where the previous DH frame sits, expressed in THIS link's own
    local coordinates (the frame this link's geometry is parented to)."""
    B = SE3(0, 0, d) * SE3(a, 0, 0) * SE3.Rx(alpha)
    return np.array(B.inv().t)

def attach_stick_geometry(robot, radius=0.025, colour=(0.5, 0.5, 0.55, 1)):
    per_link = isinstance(colour, (list, tuple)) and len(colour) == robot.n \
        and all(isinstance(c, (list, tuple)) for c in colour)

    for i, link in enumerate(robot.links):
        a, d, alpha = link.a, link.d, link.alpha
        target = _local_vector_to_previous_frame(a, d, alpha)
        length = np.linalg.norm(target)
        if length < 1e-6:
            continue

        z_axis = np.array([0, 0, 1])
        direction = target / length
        axis = np.cross(z_axis, direction)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            rot = SE3() if direction[2] > 0 else SE3.Rx(np.pi)
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rot = SE3.AngVec(angle, axis)

        link_colour = colour[i] if per_link else colour
        cyl = geometry.Cylinder(
            radius=radius, length=length,
            pose=SE3(target / 2) * rot,
            color=link_colour,
        )
        link.geometry = [cyl]
    return robot
