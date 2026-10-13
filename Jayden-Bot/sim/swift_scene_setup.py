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
from pathlib import Path

for _root in Path(__file__).resolve().parents[1:3]:
    sys.path.insert(0, str(_root))

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as geometry
from spatialmath import SE3

from collision import Box, VCylinder
from models.rebot_b601_dm import ReBotB601DM
from models.rebot_visual import ReBotVisual
from models.staubli_tx2_60 import D1, StaubliTX260

try:
    from spatialgeometry.geom.CollisionShape import CollisionShape
    if not hasattr(CollisionShape, "_update_pyb"):
        CollisionShape._update_pyb = CollisionShape._update_coal
except ImportError:
    pass

# Two fixes to Swift's page server so mesh models (the reBot URDF meshes, the
# duck can) load on Windows. Without them the browser shows "Application
# error: a client-side exception has occurred" and the sim crashes with
# ValueError: invalid literal for int() ... 'Connected'.
#  1. The browser asks for a mesh as /retrieve/C:/Users/...; Swift opens
#     "/C:/Users/...", which doesn't exist on Windows, so every mesh 404s.
#     Drop the slash in front of the drive letter.
#  2. It serves one request at a time with a 5-deep queue, and Windows resets
#     connections that overflow it while ~40 meshes load at once. Serve
#     requests in parallel with a deeper queue.
try:
    import re
    import socketserver
    import types
    from swift import SwiftRoute

    class _ParallelServer(socketserver.ThreadingTCPServer):
        daemon_threads = True
        request_queue_size = 128

        def __init__(self, address, handler, *args, **kwargs):
            class WindowsPathHandler(handler):
                def send_file_via_real_path(self):
                    if re.match(r"^/[A-Za-z]:/", self.path):
                        self.path = self.path[1:]
                    return super().send_file_via_real_path()

            super().__init__(address, WindowsPathHandler, *args, **kwargs)

    SwiftRoute.socketserver = types.SimpleNamespace(TCPServer=_ParallelServer)
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
# Turned to face the platform: the reBot's waist only turns +/-160 deg, and
# unturned the platform sits in the gap behind it.
ARM2_BASE = SE3(1.05, 0.20, TABLE_HEIGHT) * SE3.Rz(np.pi)

ARM1_PARK = np.deg2rad([150, 0, 90, 0, 0, 0])

PLATFORM_POSITION = SE3(0.65, 0.30, TABLE_HEIGHT)
PLATFORM_THICKNESS = 0.01
PLATFORM_TOP = TABLE_HEIGHT + PLATFORM_THICKNESS

CAN_HOLDER_POSITION = SE3(1.10, 0.55, TABLE_HEIGHT)
CAN_HOLDER_HEIGHT = 0.10
CAN_HOLDER_TOP = TABLE_HEIGHT + CAN_HOLDER_HEIGHT

REBOT_TOOL = None
REBOT_FINGER_OPEN = 0.025       # each finger's travel from centre when the gripper is open (m)
REBOT_URDF_PATH = str(Path(__file__).resolve().parents[1] / "models" / "rebot_description" / "urdf" / "ReBot_Arm_DM.urdf")
CAN_MESH_PATH = str(Path(__file__).resolve().parents[1] / "models" / "duck_watering_can.stl")

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
    I/N 0929940), drawn from models/duck_watering_can.stl. Dimensions are
    measured off the real can and match the mesh.

    Frame sits at the base centre of the body, z up. The body's local x
    points toward the spout (the 'front' of the duck). The handle arcs
    over the top, its bar running along local x, so the gripper comes in
    level from the side (along local y) and closes one finger over the bar
    and one under it, through the gap between the bar and the body.

    The can is very front heavy (the water sits toward the spout), which
    the simulation does not model -- on the real arm it will try to nose
    down about the handle."""

    BODY_LENGTH = 0.200             # along local x
    BODY_WIDTH = 0.125              # along local y
    BODY_HEIGHT = 0.160
    BODY_SIZE = [BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT]

    HANDLE_TOP = 0.230              # top of the handle bar above the base
    HANDLE_SPAN = 0.140             # along local x, outside of post to outside of post
    HANDLE_WIDTH = 0.020            # bar width, what the fingers close on
    HANDLE_BAR_HEIGHT = 0.220       # centre of the bar above the base
    GRIP_DROP = 0.030               # grip point below the top of the handle
    SPOUT_REACH = 0.150             # spout tip ahead of the grip point, along local x

    SPOUT_ROOT = [0.100, 0.0, 0.120]    # where the spout leaves the front of the body
    SPOUT_TIP_HEIGHT = 0.190
    SPOUT_RADIUS = 0.022

    COLOUR = [0.95, 0.80, 0.10, 1]

    def __init__(self, pose: SE3):
        self.mesh = geometry.Mesh(CAN_MESH_PATH, color=self.COLOUR)

        self.handle_grip_local = np.array([0.0, 0.0, self.HANDLE_TOP - self.GRIP_DROP])
        self.handle_bar_local = np.array([0.0, 0.0, self.HANDLE_BAR_HEIGHT])
        self.handle_bar_direction_local = np.array([1.0, 0.0, 0.0])
        self.bill_root_local = np.array(self.SPOUT_ROOT)
        self.spout_tip_local = np.array([self.SPOUT_REACH, 0.0, self.SPOUT_TIP_HEIGHT])

        self.set_pose(pose)

    @property
    def shapes(self):
        return [self.mesh]

    def set_pose(self, T: SE3):
        self.T = T
        self.mesh.T = T.A


class DroppedObject:
    """A loose object someone can drop into the workcell: a red box the arms
    must not hit. It waits on the floor in front of the desk, out of reach of
    both arms, until it is dropped. `version` counts every change, so the
    sequence knows when to re-check the motion it has already planned.

    `position` is where the object ends up and is what collision checking
    uses from the moment it is dropped; the drawn box falls onto that spot
    over the next few frames."""

    SIZE = [0.08, 0.08, 0.08]
    STOWED = [0.70, -0.30, 0.04]
    DROP_HEIGHT = 0.40      # the drawn box starts this far above where it lands
    DROP_PER_FRAME = 0.05

    def __init__(self):
        self.shape = geometry.Cuboid(scale=self.SIZE, pose=SE3(self.STOWED), color=[0.85, 0.1, 0.1, 1])
        self.position = np.array(self.STOWED)
        self.placed = False
        self.version = 0
        self._still_to_fall = 0.0

    def place(self, position, fall=True):
        self._set(position, True, self.DROP_HEIGHT if fall else 0.0)

    def stow(self):
        self._set(self.STOWED, False, 0.0)

    def update(self):
        if self._still_to_fall > 0:
            self._still_to_fall = max(0.0, self._still_to_fall - self.DROP_PER_FRAME)
            self._draw()

    def _set(self, position, placed, still_to_fall):
        self.position, self.placed = np.asarray(position, float).copy(), placed
        self._still_to_fall = still_to_fall
        self._draw()
        self.version += 1

    def _draw(self):
        self.shape.T = SE3(self.position + [0, 0, self._still_to_fall]).A


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
    rebot_visual: ReBotVisual
    can: WateringCanDuck
    pots: dict
    figures: list
    obstacles: list
    hazard: DroppedObject
    arm2_finger_opening: float = REBOT_FINGER_OPEN

    def refresh(self):
        for f in self.figures:
            f.update()
        self.rebot_visual.update(self.arm2, self.arm2.q, self.arm2_finger_opening)
        self.hazard.update()

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

    rebot_visual = ReBotVisual(REBOT_URDF_PATH, dh_tool=arm2.tool, base=arm2.base)
    rebot_visual.update(arm2, arm2.q, REBOT_FINGER_OPEN)

    can = WateringCanDuck(SE3(*CAN_HOLDER_POSITION.t[:2], CAN_HOLDER_TOP))
    pots = {p.pot_id: Pot(p, shelf_pose(p)) for p in pot_profiles}
    hazard = DroppedObject()

    figures = []
    if env is not None:
        env.add(build_table())
        for arm in (arm1, arm2):
            if arm is arm2:
                rebot_visual.add_to(env)
            elif isinstance(arm, rtb.DHRobot):
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
        env.add(hazard.shape)
        env.add(build_wall())
        env.step()

    return Scene(env, arm1, arm2, rebot_visual, can, pots, figures, static_obstacles(), hazard)


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
