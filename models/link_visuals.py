"""
Shared Visual helper for DH models and Swift scene objects.

Simple cylinders for each link
"""

import spatialgeometry as geometry
from spatialmath import SE3
from math import pi

def attach_stick_geometry(robot, radius=0.052, colour=[0.5, 0.5, 0.5, 1]):
    """Attach a simple cylinder to a robot link for visualization in Swift.
    The cylinder is aligned along the link's z-axis and extends from the
    link's origin to the next joint."""
    per_link = isinstance(colour, (list, tuple)) and len(colour) == len(colour) == robot.n and all(isinstance(c, (list, tuple)) for c in colour)
    for i, link in enumerate(robot.links): 
        link_colour = colour[i] if per_link else colour
        shapes = []
        a, d = link.a, link.d
        if abs(d) > 1e-6:
            # Cylinder along z-axis
            shapes.append(geometry.Cylinder(radius=radius, length=abs(d), pose=SE3(0, 0, d/2), color=link_colour))

        if abs(a) > 1e-6:
            # Cylinder along x-axis
            shapes.append(geometry.Cylinder(radius=radius, length=abs(a), pose=SE3(a/2, 0, d) * SE3.Ry(pi/2), color=link_colour))
        if shapes:
            link.geometry = shapes
    return robot
