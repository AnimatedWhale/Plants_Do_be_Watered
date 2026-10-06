"""
Collision checking for the ShelfCare scene.

Robot links are capsules (a line segment with a radius) and held objects are
cylinders; both are approximated by chains of spheres. Furniture and objects
standing in the scene are axis-aligned boxes or upright cylinders. Checking a
sphere against a box or cylinder is a couple of lines of maths, so every
waypoint of every planned move can be checked before the arm moves.

Distances are signed: positive = gap, negative = overlapping.
"""

from dataclasses import dataclass

import numpy as np


# Required clearances (m)
ARM_MARGIN = 0.01         # arm links vs furniture / objects
ARM_ARM_MARGIN = 0.02     # one arm vs the other
HELD_TOLERANCE = 0.003    # a held object may touch (rest on) a surface, not sink into it


# ---------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------

class Box:
    """Axis-aligned box."""

    def __init__(self, name, centre, size):
        self.name = name
        self.c = np.asarray(centre, float)
        self.h = np.asarray(size, float) / 2

    def distance(self, pts):
        q = np.abs(pts - self.c) - self.h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside


class VCylinder:
    """Upright cylinder standing on `base` (bottom centre)."""

    def __init__(self, name, base, radius, height):
        self.name = name
        self.b = np.asarray(base, float)
        self.r = radius
        self.half = height / 2

    def distance(self, pts):
        dr = np.linalg.norm(pts[:, :2] - self.b[:2], axis=1) - self.r
        dz = np.abs(pts[:, 2] - (self.b[2] + self.half)) - self.half
        outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(dr, dz), 0.0)
        return outside + inside


# ---------------------------------------------------------------------
# Sphere chains for moving things
# ---------------------------------------------------------------------

@dataclass
class Spheres:
    name: str
    centres: np.ndarray   # (N, 3)
    radii: np.ndarray     # (N,)
    gripper: bool = False  # part of the gripper (allowed to reach into the object it's working on)


def capsule(name, p0, p1, radius, gripper=False) -> Spheres:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / radius)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return Spheres(name, p0 + t * (p1 - p0), np.full(n, radius), gripper)


def cylinder(name, T, radius, z0, z1) -> Spheres:
    """Cylinder along the local z axis of pose T, from z0 to z1, as spheres
    whose ends sit flush with the cylinder's end faces on its axis."""
    rs = min(radius, (z1 - z0) / 2)
    n = max(1, int(np.ceil((z1 - z0 - 2 * rs) / (radius / 2))) + 1)
    z = np.linspace(z0 + rs, z1 - rs, n)
    pts = (T.A @ np.vstack([np.zeros((2, n)), z, np.ones(n)]))[:3].T
    return Spheres(name, pts, np.full(n, rs))


def clearance(a: Spheres, obstacle) -> float:
    """Smallest gap between a sphere chain and a Box / VCylinder / Spheres."""
    if isinstance(obstacle, Spheres):
        d = np.linalg.norm(a.centres[:, None, :] - obstacle.centres[None, :, :], axis=2)
        return float((d - a.radii[:, None] - obstacle.radii[None, :]).min())
    return float((obstacle.distance(a.centres) - a.radii).min())


# ---------------------------------------------------------------------
# Robot bodies
# ---------------------------------------------------------------------

def link_capsules(robot, q, radii, last_frame):
    """Capsules between consecutive DH frame origins from frame 1 to
    `last_frame` (the fixed base column is skipped).

    radii[i] is the radius of the link from frame i+1 to frame i+2."""
    pts = [T.t for T in robot.fkine_all(q)]
    return [
        capsule(f"{robot.name} link {i + 1}", pts[i], pts[i + 1], radii[i - 1])
        for i in range(1, last_frame)
        if np.linalg.norm(pts[i + 1] - pts[i]) > 1e-3
    ]


def tx2_60_body(dh_twin, q, gripper_length, finger_offset):
    """TX2-60 links up to the flange, then the palm (a bar across tool y) and
    two fingers, matching _gripper_geometry in the model."""
    parts = link_capsules(dh_twin, q, radii=[0.06, 0.06, 0.05, 0.04, 0.04], last_frame=6)
    tcp = dh_twin.fkine(q)
    z, y = tcp.R[:, 2], tcp.R[:, 1]
    flange = tcp.t - gripper_length * z
    palm = flange + 0.015 * z
    parts.append(capsule(f"{dh_twin.name} palm", palm - finger_offset * y, palm + finger_offset * y,
                         0.025, gripper=True))
    for side in (1, -1):
        parts.append(capsule(f"{dh_twin.name} finger", flange + 0.03 * z + side * finger_offset * y,
                             tcp.t + 0.02 * z + side * finger_offset * y, 0.015, gripper=True))
    return parts


def rebot_body(robot, q):
    """reBot links; its last DH link (wrist to TCP) holds the gripper, so it's
    tagged as gripper and stops 6 cm short of the TCP."""
    parts = link_capsules(robot, q, radii=[0.035] * 4, last_frame=5)
    wrist = robot.fkine_all(q)[5].t
    tcp = robot.fkine(q)
    parts.append(capsule(f"{robot.name} gripper", wrist, tcp.t - 0.06 * tcp.R[:, 2], 0.035, gripper=True))
    return parts


# ---------------------------------------------------------------------
# One check
# ---------------------------------------------------------------------

def first_collision(moving, others, held=None, obstacles=(), work_object=None):
    """Return a description of the first collision, or None.

    moving       the moving arm's parts (list of Spheres)
    others       the other arm's parts (list of Spheres)
    held         the object the moving arm holds (Spheres), if any
    obstacles    [(name, Box | VCylinder)] furniture + objects not held
    work_object  name of the obstacle the moving arm is working on (approaching,
                 just released): the gripper may reach into it, and the rest of
                 the arm may come close to it but not touch it
    """
    for part in moving:
        for name, obs in obstacles:
            if part.gripper and name == work_object:
                continue
            margin = 0.0 if name == work_object else ARM_MARGIN
            gap = clearance(part, obs)
            if gap < margin:
                return f"{part.name} hits {name} ({_depth(gap, margin)})"
        for other in others:
            gap = clearance(part, other)
            if gap < ARM_ARM_MARGIN:
                return f"{part.name} hits {other.name} ({_depth(gap, ARM_ARM_MARGIN)})"

    if held is not None:
        for name, obs in obstacles:
            gap = clearance(held, obs)
            if gap < -HELD_TOLERANCE:
                return f"held {held.name} hits {name} ({_depth(gap, 0)})"
        for part in [p for p in moving if not p.gripper] + list(others):
            gap = clearance(held, part)
            if gap < ARM_MARGIN:
                return f"held {held.name} hits {part.name} ({_depth(gap, ARM_MARGIN)})"
    return None


def _depth(gap, margin):
    return f"{-gap * 1000:.0f} mm overlap" if gap < 0 else f"only {gap * 1000:.0f} mm clear, need {margin * 1000:.0f}"
