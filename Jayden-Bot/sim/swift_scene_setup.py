"""
ShelfCare Swift scene: the room with its window sill and plants, the desk, both
arms, the platform, the watering can and the safety equipment.

build_scene() is shared with Selfcare_sequence.py so both use one layout.
To just look at the scene:
    python Jayden-Bot/sim/swift_scene_setup.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

for _root in Path(__file__).resolve().parents[1:3]:
    sys.path.insert(0, str(_root))

from importlib.metadata import PackageNotFoundError, version

# Older toolboxes break this sim in confusing ways: roboticstoolbox 1.3 puts
# the TX2-60's gripper 150 mm out (the arm can't reach the pots), and looks
# for the reBot's meshes in the wrong folder, so the Swift page crashes and
# Python prints a websockets "ConnectionClosedError". Usually it means the
# wrong Python environment was picked, so stop with a clear message instead.
_NEEDED = {"roboticstoolbox-python": (1, 4), "spatialgeometry": (1, 4)}
_too_old = []
for _package, _minimum in _NEEDED.items():
    try:
        _have = version(_package)
    except PackageNotFoundError:
        _have = "not installed"
    if _have == "not installed" or tuple(int(p) for p in _have.split(".")[:2]) < _minimum:
        _too_old.append(f"  {_package}: need {'.'.join(map(str, _minimum))} or newer, this Python has {_have}")
if _too_old:
    sys.exit("This Python environment's packages are too old for the ShelfCare sim:\n" + "\n".join(_too_old)
             + f"\n(running {sys.executable})\n"
             "Make a fresh environment for this project (upgrading an old one in place can leave\n"
             "spatialgeometry half-replaced), from the Plants_Do_be_Watered folder:\n"
             "  py -3.11 -m venv C:\\venvs\\shelfcare\n"
             "  C:\\venvs\\shelfcare\\Scripts\\python.exe -m pip install -r requirements.txt\n"
             "then run the sim with C:\\venvs\\shelfcare\\Scripts\\python.exe (or pick it in VS Code).")

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

# Fixes Swift's page server on Windows so the meshes load: drop the slash before the
# drive letter in mesh paths (else 404), and serve requests in parallel (else resets).
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

    # swift 1.1 asks for TCPServer, swift 2.x for ThreadingTCPServer: give it both
    SwiftRoute.socketserver = types.SimpleNamespace(TCPServer=_ParallelServer,
                                                    ThreadingTCPServer=_ParallelServer)
except ImportError:
    pass


TABLE_HEIGHT = 0.75
TABLE_SIZE = [1.4, 0.9, 0.03]      # 0.9 m deep so arm 1 can sit back from the sill
TABLE_CENTRE = SE3(0.7, 0.25, TABLE_HEIGHT - TABLE_SIZE[2] / 2)

# The room: one long wall behind the desk with a wide window in it. The plants
# sit on the window sill (still called the shelf in the code).
WALL_Y = 0.75
ROOM_X = (-1.5, 2.9)                # the wall runs this far along x
ROOM_HEIGHT = 2.4
WALL_SIZE = [ROOM_X[1] - ROOM_X[0], 0.02, ROOM_HEIGHT]
WALL_CENTRE = SE3(sum(ROOM_X) / 2, WALL_Y + WALL_SIZE[1] / 2, ROOM_HEIGHT / 2)
FLOOR_DEPTH = 3.0                   # how far the floor runs out from the wall
FLOOR_TOP = 0.01                    # just above Swift's own ground grid, so they don't flicker
SHELF_HEIGHT = 0.95                # lowered from 1.05 so arm 1 reaches the plants without being cramped
SHELF_SIZE = [0.70, 0.20, 0.02]
SHELF_CENTRE = SE3(0.25, WALL_Y - SHELF_SIZE[1] / 2, SHELF_HEIGHT - SHELF_SIZE[2] / 2)
WINDOW_X = (-0.6, 1.4)              # the window opening, behind both arms
WINDOW_Z = (SHELF_HEIGHT, 1.85)     # its bottom is the sill the plants sit on

ARM1_BASE = SE3(0.25, -0.02, TABLE_HEIGHT)   # set back from the shelf (was y = 0.12): more room to move
# turned 180 deg: the reBot's waist only turns +/-160 deg, and the platform is behind it
ARM2_BASE = SE3(1.05, 0.20, TABLE_HEIGHT) * SE3.Rz(np.pi)

ARM1_PARK = np.deg2rad([150, 0, 90, 0, 0, 0])

PLATFORM_POSITION = SE3(0.65, 0.30, TABLE_HEIGHT)
PLATFORM_THICKNESS = 0.01
PLATFORM_TOP = TABLE_HEIGHT + PLATFORM_THICKNESS

CAN_HOLDER_POSITION = SE3(1.10, 0.55, TABLE_HEIGHT)
CAN_HOLDER_HEIGHT = 0.10
CAN_HOLDER_TOP = TABLE_HEIGHT + CAN_HOLDER_HEIGHT

REBOT_TOOL = None
REBOT_FINGER_OPEN = 0.016       # each finger's travel from centre when open (m); fits the gap under the can handle
REBOT_URDF_DIR = Path(__file__).resolve().parents[1] / "models" / "rebot_description" / "urdf"
REBOT_URDF_FULL = str(REBOT_URDF_DIR / "ReBot_Arm_DM.urdf")
REBOT_URDF_LITE = str(REBOT_URDF_DIR / "ReBot_Arm_DM_lite.urdf")    # simplified meshes, for slower computers
REBOT_URDF_PATH = REBOT_URDF_FULL       # run_lite.py switches this to REBOT_URDF_LITE
CAN_MESH_PATH = str(Path(__file__).resolve().parents[1] / "models" / "duck_watering_can.stl")

# Safety gear. The light curtains sit just outside the space the arms sweep through.
# Sized from both arms' full reach (gripper over every joint angle: x -0.57..1.79, y down to -0.83)
# plus 0.2 m for a held pot or can, so neither arm can reach out of the zone.
GUARD_X = (-0.80, 2.00)         # left and right light curtains
CURTAIN_Y = -1.05               # front light curtain
GUARD_HEIGHT = 1.80
GUARD_BACK_Y = WALL_Y - 0.02    # the back posts stand against the wall
HIDDEN = SE3(0, 0, -20)         # where a shape goes when it should not be seen
YELLOW = [0.95, 0.8, 0.1, 1]


def build_table():
    return geometry.Cuboid(scale=TABLE_SIZE, pose=TABLE_CENTRE, color=[0.75, 0.6, 0.45, 1])


def _box(x0, x1, y0, y1, z0, z1, colour):
    """A box given by its x, y and z ranges."""
    return geometry.Cuboid(scale=[x1 - x0, y1 - y0, z1 - z0],
                           pose=SE3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), color=colour)


def build_room():
    """Floor, desk legs, and the wall with its window (frame only, so it's see-through)."""
    paint, frame = [0.93, 0.91, 0.86, 1], [0.97, 0.97, 0.95, 1]
    (x0, x1), (wx0, wx1), (wz0, wz1) = ROOM_X, WINDOW_X, WINDOW_Z
    y0, y1 = WALL_Y, WALL_Y + WALL_SIZE[1]
    shapes = [
        _box(x0, x1, WALL_Y - FLOOR_DEPTH, y1, 0.0, FLOOR_TOP, [0.62, 0.62, 0.60, 1]),     # floor
        # wall, in four pieces around the window opening
        _box(x0, x1, y0, y1, 0.0, wz0, paint),
        _box(x0, x1, y0, y1, wz1, ROOM_HEIGHT, paint),
        _box(x0, wx0, y0, y1, wz0, wz1, paint),
        _box(wx1, x1, y0, y1, wz0, wz1, paint),
        # window frame: bottom rail, head, two jambs, two mullions
        _box(wx0, wx1, y0, y1 + 0.01, wz0, wz0 + 0.03, frame),
        _box(wx0, wx1, y0, y1 + 0.01, wz1 - 0.05, wz1, frame),
        _box(wx0, wx0 + 0.05, y0, y1 + 0.01, wz0, wz1, frame),
        _box(wx1 - 0.05, wx1, y0, y1 + 0.01, wz0, wz1, frame),
        _box(wx0 + (wx1 - wx0) / 3 - 0.02, wx0 + (wx1 - wx0) / 3 + 0.02, y0, y1 + 0.01, wz0, wz1, frame),
        _box(wx0 + 2 * (wx1 - wx0) / 3 - 0.02, wx0 + 2 * (wx1 - wx0) / 3 + 0.02, y0, y1 + 0.01, wz0, wz1, frame),
    ]
    # desk legs, so the desk stands on the floor
    leg = 0.05
    tx0, tx1 = TABLE_CENTRE.t[0] - TABLE_SIZE[0] / 2, TABLE_CENTRE.t[0] + TABLE_SIZE[0] / 2
    ty0, ty1 = TABLE_CENTRE.t[1] - TABLE_SIZE[1] / 2, TABLE_CENTRE.t[1] + TABLE_SIZE[1] / 2
    for lx in (tx0 + 0.05, tx1 - 0.05 - leg):
        for ly in (ty0 + 0.05, ty1 - 0.05 - leg):
            shapes.append(_box(lx, lx + leg, ly, ly + leg, FLOOR_TOP, TABLE_HEIGHT - TABLE_SIZE[2], [0.45, 0.35, 0.25, 1]))
    return shapes


def build_shelf():
    """The window sill the plants sit on: a painted board level with the bottom of the window."""
    return geometry.Cuboid(scale=SHELF_SIZE, pose=SHELF_CENTRE, color=[0.96, 0.96, 0.93, 1])


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
    """Potted plant (pot + foliage). Its frame is the bottom centre of the pot."""

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
    """Duck watering can, drawn from duck_watering_can.stl (sizes measured off the real can).
    Frame: bottom centre of the body, z up, x toward the spout."""

    # The STL is in mm with its origin off to one side; this puts it in metres on the can frame.
    MESH_SCALE = 0.001
    MESH_OFFSET = SE3(0.0, 0.0625, 0.050)

    BODY_LENGTH = 0.200             # along local x, not counting the bill
    BODY_WIDTH = 0.110              # along local y
    BODY_HEIGHT = 0.160             # up to the top of the head
    BODY_SIZE = [BODY_LENGTH, BODY_WIDTH, BODY_HEIGHT]

    HANDLE_TOP = 0.150              # top of the handle bar above the base
    HANDLE_WIDTH = 0.020            # bar thickness, what the fingers close on
    HANDLE_BAR_HEIGHT = 0.140       # centre of the bar above the base
    GRIP_X = -0.055                 # where along the bar the gripper holds it (biggest gap under the bar)
    GRIP_DROP = 0.030               # measured grip point, below the top of the handle

    SPOUT_ROOT = [0.100, 0.0, 0.120]    # where the bill leaves the head
    SPOUT_TIP = [0.120, 0.0, 0.120]     # end of the bill

    COLOUR = [0.95, 0.80, 0.10, 1]

    def __init__(self, pose: SE3):
        self.mesh = geometry.Mesh(CAN_MESH_PATH, scale=[self.MESH_SCALE] * 3, color=self.COLOUR)

        self.handle_grip_local = np.array([self.GRIP_X, 0.0, self.HANDLE_TOP - self.GRIP_DROP])
        self.handle_bar_local = np.array([self.GRIP_X, 0.0, self.HANDLE_BAR_HEIGHT])
        self.handle_bar_direction_local = np.array([1.0, 0.0, 0.0])
        self.bill_root_local = np.array(self.SPOUT_ROOT)
        self.spout_tip_local = np.array(self.SPOUT_TIP)

        self.set_pose(pose)

    @property
    def shapes(self):
        return [self.mesh]

    def set_pose(self, T: SE3):
        self.T = T
        self.mesh.T = (T * self.MESH_OFFSET).A


class DroppedObject:
    """Red test box dropped into the cell. `version` changes whenever it moves."""

    SIZE = [0.08, 0.08, 0.08]
    STOWED = [0.70, -1.20, FLOOR_TOP + 0.04]       # on the floor outside the light curtain
    DROP_HEIGHT = 0.40      # falls from this high
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


def guard_boxes():
    """The four light curtain posts as (name, centre, size)."""
    return [("light curtain post", [x, y, GUARD_HEIGHT / 2], [0.04, 0.04, GUARD_HEIGHT])
            for x in GUARD_X for y in (CURTAIN_Y, GUARD_BACK_Y)]


class SafetyEquipment:
    """Safety gear from the risk assessment:
      - light curtains on the three open sides (the wall closes the back):
        sense anyone reaching in, which pauses the arms
      - signal tower: shows the cell state (green run, amber paused, red stopped)
      - e-stop station: a stop button outside the cell, within reach
      - yellow markings: keep-out area at the platform, stand-back lines on the floor
    """

    LAMPS = {   # name: (height above the post, lit colour, unlit colour)
        "green": (0.13, [0.1, 0.9, 0.2, 1], [0.08, 0.3, 0.1, 1]),
        "amber": (0.20, [1.0, 0.7, 0.0, 1], [0.35, 0.27, 0.05, 1]),
        "red": (0.27, [1.0, 0.1, 0.1, 1], [0.35, 0.1, 0.1, 1]),
    }
    BEAM_HEIGHTS = np.linspace(0.85, 1.70, 6)

    def __init__(self):
        left, right = GUARD_X
        mid_x, width = (left + right) / 2, right - left

        self.fixed = [geometry.Cuboid(scale=size, pose=SE3(centre), color=YELLOW)
                      for _, centre, size in guard_boxes()]
        # stand-back lines on the floor around the three open sides, and the keep-out disc under the platform
        line_y0 = CURTAIN_Y - 0.30
        self.fixed.append(_box(left - 0.30, right + 0.30, line_y0 - 0.025, line_y0 + 0.025, FLOOR_TOP, FLOOR_TOP + 0.004, YELLOW))
        for x in (left - 0.30, right + 0.30):
            self.fixed.append(_box(x - 0.025, x + 0.025, line_y0, GUARD_BACK_Y, FLOOR_TOP, FLOOR_TOP + 0.004, YELLOW))
        self.fixed.append(geometry.Cylinder(radius=0.20, length=0.002,
                                            pose=SE3(*PLATFORM_POSITION.t[:2], TABLE_HEIGHT + 0.001), color=YELLOW))
        # e-stop station on the outside of the right post
        self.fixed.append(geometry.Cuboid(scale=[0.10, 0.06, 0.10], pose=SE3(right, CURTAIN_Y - 0.05, 1.0),
                                          color=YELLOW))
        self.fixed.append(geometry.Cylinder(radius=0.03, length=0.03, color=[0.9, 0.05, 0.05, 1],
                                            pose=SE3(right, CURTAIN_Y - 0.095, 1.0) * SE3.Rx(np.pi / 2)))
        # signal tower pole on top of the right post
        self.fixed.append(geometry.Cylinder(radius=0.012, length=0.10, color=[0.2, 0.2, 0.2, 1],
                                            pose=SE3(right, CURTAIN_Y, GUARD_HEIGHT + 0.05)))

        # lamps and beams are drawn twice (lit and unlit); the unused one is hidden
        self.lamps = {}
        for name, (height, lit, unlit) in self.LAMPS.items():
            pose = SE3(right, CURTAIN_Y, GUARD_HEIGHT + height)
            self.lamps[name] = (geometry.Cylinder(radius=0.035, length=0.06, color=lit),
                                geometry.Cylinder(radius=0.035, length=0.06, color=unlit), pose)
        # beams on the front (along x) and both sides (along y)
        depth, mid_y = GUARD_BACK_Y - CURTAIN_Y, (GUARD_BACK_Y + CURTAIN_Y) / 2
        runs = [(SE3(mid_x, CURTAIN_Y, 0) * SE3.Ry(np.pi / 2), width)]
        runs += [(SE3(x, mid_y, 0) * SE3.Rx(np.pi / 2), depth) for x in GUARD_X]
        self.beams = []
        for start, length in runs:
            for z in self.BEAM_HEIGHTS:
                pose = SE3(0, 0, z) * start
                self.beams.append((geometry.Cylinder(radius=0.008, length=length, color=[1.0, 0.0, 0.0, 1]),
                                   geometry.Cylinder(radius=0.004, length=length, color=[0.9, 0.2, 0.2, 0.35]),
                                   pose))
        # someone's arm reaching in, shown while the curtain is broken
        self.intruder = geometry.Cuboid(scale=[0.08, 0.30, 0.06], color=[0.9, 0.7, 0.55, 1])
        self.intruder_pose = SE3(mid_x, CURTAIN_Y, 1.0)

        self._shown = None
        self.show("amber")

    @property
    def shapes(self):
        changing = [shape for lit, unlit, _ in list(self.lamps.values()) + self.beams for shape in (lit, unlit)]
        return self.fixed + changing + [self.intruder]

    def show(self, signal, zone_breached=False):
        """Light one lamp and show whether the light curtain is broken."""
        if (signal, zone_breached) == self._shown:
            return
        self._shown = (signal, zone_breached)
        for name, (lit, unlit, pose) in self.lamps.items():
            lit.T = (pose if name == signal else HIDDEN).A
            unlit.T = (HIDDEN if name == signal else pose).A
        for lit, unlit, pose in self.beams:
            lit.T = (pose if zone_breached else HIDDEN).A
            unlit.T = (HIDDEN if zone_breached else pose).A
        self.intruder.T = (self.intruder_pose if zone_breached else HIDDEN).A


def _segment_pose(p0, p1):
    z = (p1 - p0) / np.linalg.norm(p1 - p0)
    ref = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0]
    o = np.cross(z, ref)
    return SE3((p0 + p1) / 2) * SE3.OA(o / np.linalg.norm(o), z)


class StickFigure:
    """Draws a DH robot as cylinders between its joint frames (Swift can't draw a DHRobot)."""

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
    ] + [(name, Box(name, centre, size)) for name, centre, size in guard_boxes()]


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
    safety: SafetyEquipment
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
    safety = SafetyEquipment()

    figures = []
    if env is not None:
        for shape in build_room():
            env.add(shape)
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
        for shape in safety.shapes:
            env.add(shape)
        if hasattr(env, "set_camera_pose"):
            env.set_camera_pose([0.6, -3.2, 2.3], [0.6, 0.0, 0.9])     # from in front of the zone, looking at the desk
        env.step()

    return Scene(env, arm1, arm2, rebot_visual, can, pots, figures, static_obstacles(), hazard, safety)


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
