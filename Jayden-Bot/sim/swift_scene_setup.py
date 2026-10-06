"""
ShelfCare Swift scene setup.

Two robots on one desk, each object at a fixed (predetermined) location:
  - Stäubli TX2-60 (arm 1), bolted to the desk under a wall shelf
  - reBot B601-DM (arm 2), bolted to the other end of the desk
  - wall shelf of potted plants, above/behind arm 1
  - handover platform on the desk, reachable by both arms
  - watering-can holder beside arm 2, with the can on it

Story: arm 1 lifts a plant off the shelf and puts it on the platform; arm 2
picks up the watering can, waters the plant and puts the can back; arm 1
returns the plant to its shelf slot.

build_scene() is shared with Selfcare_sequence.py so both use one layout.
Run this file on its own just to look at the scene:
    python Jayden-Bot/sim/swift_scene_setup.py
"""

import sys
from dataclasses import dataclass
from math import pi
from pathlib import Path

for _root in Path(__file__).resolve().parents[1:3]:
    sys.path.insert(0, str(_root))

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as geometry
from spatialmath import SE3

from collision import Box, VCylinder
from models.rebot_b601_dm import ReBotB601DM
from models.staubli_tx2_60 import D1, StaubliTX260

try:
    from spatialgeometry.geom.CollisionShape import CollisionShape
    if not hasattr(CollisionShape, "_update_pyb"):
        CollisionShape._update_pyb = CollisionShape._update_coal
except ImportError:
    pass


TABLE_HEIGHT = 0.75
TABLE_SIZE = [1.4, 0.7, 0.03]
TABLE_CENTRE = SE3(0.7, 0.35, TABLE_HEIGHT - TABLE_SIZE[2] / 2)

WALL_Y = 0.75
WALL_SIZE = [1.4, 0.02, 1.2]
WALL_CENTRE = SE3(0.7, WALL_Y + WALL_SIZE[1] / 2, TABLE_HEIGHT + WALL_SIZE[2] / 2)
SHELF_HEIGHT = 1.05
SHELF_SIZE = [0.70, 0.20, 0.02]
SHELF_CENTRE = SE3(0.25, WALL_Y - SHELF_SIZE[1] / 2, SHELF_HEIGHT - SHELF_SIZE[2] / 2)

ARM1_BASE = SE3(0.25, 0.12, TABLE_HEIGHT)
ARM2_BASE = SE3(1.05, 0.20, TABLE_HEIGHT)

ARM1_PARK = np.deg2rad([150, 0, 90, 0, 0, 0])

PLATFORM_POSITION = SE3(0.65, 0.30, TABLE_HEIGHT)
PLATFORM_THICKNESS = 0.01
PLATFORM_TOP = TABLE_HEIGHT + PLATFORM_THICKNESS

CAN_HOLDER_POSITION = SE3(1.10, 0.55, TABLE_HEIGHT)
CAN_HOLDER_HEIGHT = 0.10
CAN_HOLDER_TOP = TABLE_HEIGHT + CAN_HOLDER_HEIGHT

REBOT_TOOL = None


def build_table():
    return geometry.Cuboid(scale=TABLE_SIZE, pose=TABLE_CENTRE, color=[0.75, 0.6, 0.45, 1])


def build_wall():
    return geometry.Cuboid(scale=WALL_SIZE, pose=WALL_CENTRE, color=[0.85, 0.85, 0.9, 0.25])


def build_shelf():
    return geometry.Cuboid(scale=SHELF_SIZE, pose=SHELF_CENTRE, color=[0.9, 0.9, 0.88, 1])


def build_platform():
    return geometry.Cylinder(
        radius=0.10, length=PLATFORM_THICKNESS,
        pose=PLATFORM_POSITION * SE3(0, 0, PLATFORM_THICKNESS / 2),
        color=[0.6, 0.6, 0.6, 1],
    )


def build_can_holder():
    return geometry.Cylinder(
        radius=0.06, length=CAN_HOLDER_HEIGHT,
        pose=CAN_HOLDER_POSITION * SE3(0, 0, CAN_HOLDER_HEIGHT / 2),
        color=[0.2, 0.5, 0.8, 1],
    )


class Pot:
    """Potted plant: a cylinder pot with a sphere of foliage. Its frame sits
    at the bottom centre of the pot, z up, so its pose is where it stands."""

    def __init__(self, profile, pose: SE3):
        self.profile = profile
        r, h = (profile.diameter_mm / 1000) / 2, profile.height_mm / 1000
        leaf_r = 0.7 * r
        self.pot = geometry.Cylinder(radius=r, length=h, color=[0.72, 0.36, 0.2, 1])
        self.leaves = geometry.Sphere(radius=leaf_r, color=[0.3, 0.6, 0.3, 1])
        self._pot_local = SE3(0, 0, h / 2)
        self._leaves_local = SE3(0, 0, h + 0.3 * leaf_r)
        self.set_pose(pose)

    @property
    def shapes(self):
        return [self.pot, self.leaves]

    def set_pose(self, T: SE3):
        self.T = T
        self.pot.T = (T * self._pot_local).A
        self.leaves.T = (T * self._leaves_local).A


class WateringCanDuck:
    """Garden Basics 1.6L Yellow Duck Plastic Watering Can (Bunnings,
    I/N 0929940). No published dimensions -- these are estimated, sized
    to roughly match the stated 1.6 L capacity. Measure the real can and
    update when you have it.

    Frame sits at the base centre of the body, z up. The body's local x
    points toward the bill (the 'front' of the duck). The handle arcs
    over the top, its bar running along local x, so the gripper grasps
    it from directly above with fingers closing along local y."""

    BODY_RADIUS = 0.065
    BODY_HEIGHT = 0.14
    BILL_LENGTH = 0.05
    BILL_ANGLE = pi / 12

    HANDLE_HEIGHT = 0.08
    HANDLE_SPAN = 0.10
    HANDLE_BAR_RADIUS = 0.008

    COLOUR = [0.95, 0.80, 0.10, 1]

    def __init__(self, pose: SE3):
        self.body = geometry.Cylinder(radius=self.BODY_RADIUS, length=self.BODY_HEIGHT,
                                       color=self.COLOUR)
        self.bill = geometry.Cylinder(radius=0.012, length=self.BILL_LENGTH,
                                       color=[0.9, 0.5, 0.1, 1])
        self.handle = geometry.Cylinder(radius=self.HANDLE_BAR_RADIUS, length=self.HANDLE_SPAN,
                                         color=[0.3, 0.3, 0.3, 1])

        bill_root = np.array([self.BODY_RADIUS, 0.0, self.BODY_HEIGHT * 0.6])
        bill_axis = np.array([np.cos(self.BILL_ANGLE), 0.0, np.sin(self.BILL_ANGLE)])
        self._bill_local = SE3(bill_root + bill_axis * self.BILL_LENGTH / 2) * SE3.Ry(-self.BILL_ANGLE)
        self.spout_tip_local = bill_root + bill_axis * self.BILL_LENGTH
        self.bill_root_local = bill_root

        self._handle_local = SE3(0, 0, self.BODY_HEIGHT + self.HANDLE_HEIGHT) * SE3.Ry(pi / 2)
        self.handle_grip_local = np.array([0.0, 0.0, self.BODY_HEIGHT + self.HANDLE_HEIGHT])
        self.handle_bar_direction_local = np.array([1.0, 0.0, 0.0])

        self.set_pose(pose)

    @property
    def shapes(self):
        return [self.body, self.bill, self.handle]

    def set_pose(self, T: SE3):
        self.T = T
        self.body.T = T.A
        self.bill.T = (T * self._bill_local).A
        self.handle.T = (T * self._handle_local).A


def _segment_pose(p0, p1):
    z = (p1 - p0) / np.linalg.norm(p1 - p0)
    ref = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0]
    o = np.cross(z, ref)
    return SE3((p0 + p1) / 2) * SE3.OA(o / np.linalg.norm(o), z)


class StickFigure:
    """Swift can't render an rtb.DHRobot (it has no link geometry), so this
    draws one as cylinders between its joint frames, in world coordinates.
    Call update() after changing robot.q."""

    def __init__(self, robot, radius=0.022, color=(0.3, 0.45, 0.8, 1), joint_color=(0.2, 0.2, 0.25, 1)):
        self.robot = robot
        pts = self._points(robot.q)
        self.links = [
            (i, geometry.Cylinder(radius=radius, length=np.linalg.norm(pts[i + 1] - pts[i]), color=list(color)))
            for i in range(len(pts) - 1)
            if np.linalg.norm(pts[i + 1] - pts[i]) > 1e-4
        ]
        self.joints = [geometry.Sphere(radius=radius * 1.3, color=list(joint_color)) for _ in pts]
        self.update()

    def _points(self, q):
        return [T.t for T in self.robot.fkine_all(q)] + [self.robot.fkine(q).t]

    @property
    def shapes(self):
        return [cyl for _, cyl in self.links] + self.joints

    def update(self):
        pts = self._points(self.robot.q)
        for i, cyl in self.links:
            cyl.T = _segment_pose(pts[i], pts[i + 1]).A
        for p, sphere in zip(pts, self.joints):
            sphere.T = SE3(p).A


def static_obstacles():
    return [
        ("desk", Box("desk", TABLE_CENTRE.t, TABLE_SIZE)),
        ("wall", Box("wall", WALL_CENTRE.t, WALL_SIZE)),
        ("shelf", Box("shelf", SHELF_CENTRE.t, SHELF_SIZE)),
        ("platform", VCylinder("platform", PLATFORM_POSITION.t, 0.10, PLATFORM_THICKNESS)),
        ("can holder", VCylinder("can holder", CAN_HOLDER_POSITION.t, 0.06, CAN_HOLDER_HEIGHT)),
    ]


@dataclass
class Scene:
    env: object
    arm1: StaubliTX260
    arm2: rtb.Robot
    can: WateringCanDuck
    pots: dict
    figures: list
    obstacles: list

    def refresh(self):
        for f in self.figures:
            f.update()


def shelf_pose(profile) -> SE3:
    x, y = profile.shelf_position
    return SE3(x, y, SHELF_HEIGHT)


def build_scene(env=None, pot_profiles=()) -> Scene:
    arm1 = StaubliTX260()
    arm1.base = ARM1_BASE
    arm1.q = ARM1_PARK

    arm2 = ReBotB601DM()
    arm2.base = ARM2_BASE
    if REBOT_TOOL is not None:
        arm2.tool = REBOT_TOOL
    arm2.q = np.zeros(arm2.n)

    can = WateringCanDuck(SE3(*CAN_HOLDER_POSITION.t[:2], CAN_HOLDER_TOP + WateringCanDuck.BODY_HEIGHT / 2))
    pots = {p.pot_id: Pot(p, shelf_pose(p)) for p in pot_profiles}

    figures = []
    if env is not None:
        env.add(build_table())
        for arm in (arm1, arm2):
            if isinstance(arm, rtb.DHRobot):
                figure = StickFigure(arm)
                figures.append(figure)
                for shape in figure.shapes:
                    env.add(shape)
            else:
                env.add(arm)
        env.add(build_shelf())
        env.add(build_platform())
        env.add(build_can_holder())
        for shape in can.shapes + [s for pot in pots.values() for s in pot.shapes]:
            env.add(shape)
        env.add(build_wall())
        env.step()

    return Scene(env, arm1, arm2, can, pots, figures, static_obstacles())


def reach_report(pot_profiles=()):
    shoulder = (ARM1_BASE * SE3(0, 0, D1)).t
    print("Arm 1 (TX2-60, 0.670 m reach from the shoulder to the flange):")
    for p in pot_profiles:
        print(f"  shoulder -> {p.pot_id:<11}: {np.linalg.norm(shelf_pose(p).t - shoulder):.3f} m")
    print(f"  shoulder -> platform   : {np.linalg.norm(PLATFORM_POSITION.t - shoulder):.3f} m")
    print("Arm 2 (reBot), horizontal distance from its base:")
    print(f"  base -> platform  : {np.linalg.norm((PLATFORM_POSITION.t - ARM2_BASE.t)[:2]):.3f} m")
    print(f"  base -> can holder: {np.linalg.norm((CAN_HOLDER_POSITION.t - ARM2_BASE.t)[:2]):.3f} m")


def main():
    import swift
    from Selfcare_sequence import POT_REGISTRY

    env = swift.Swift()
    env.launch(realtime=True)
    build_scene(env, POT_REGISTRY.values())

    print("Scene loaded. Check the Swift tab in your browser.")
    print(f"Desk top at z={TABLE_HEIGHT} m, both arm bases mounted on it")
    print(f"Arm 1 (TX2-60) base: {ARM1_BASE.t}")
    print(f"Arm 2 (reBot) base: {ARM2_BASE.t}")
    print(f"Platform: {PLATFORM_POSITION.t}, can holder: {CAN_HOLDER_POSITION.t}")
    reach_report(POT_REGISTRY.values())

    env.hold()


if __name__ == "__main__":
    main()
