"""
ShelfCare Swift scene setup.

two robots, shelf, platform, watering can holder.
"""
import swift
import spatialgeometry as geometry
from spatialmath import SE3
from math import pi
import numpy as np
import roboticstoolbox as rtb

from models.rebot_b601_dm import ReBotB601DM
from models.link_visuals import attach_stick_geometry


# Compatibility fix (from Rudra): swift-sim 1.1 calls shape._update_pyb(),
# which spatialgeometry 1.4 renamed to _update_coal(). Alias it so Swift
# runs correctly with the current spatialgeometry version.
try:
    from spatialgeometry.geom.CollisionShape import CollisionShape
    if not hasattr(CollisionShape, "_update_pyb"):
        CollisionShape._update_pyb = CollisionShape._update_coal
except ImportError:
    pass
# ---------------------------------------------------------------------
# Office layout: one large table, both arm bases mounted on its surface,
# a window sill behind/above the table.
# TODO: adjust these once the desk/window layout is measured
# ---------------------------------------------------------------------

TABLE_HEIGHT = 0.75        # height of the table's CENTRE (not the top surface)
TABLE_SIZE = [1.4, 0.7, 0.03]   # length (x) x depth (y) x thickness
TABLE_CENTRE = SE3(0.7, 0.35, TABLE_HEIGHT)
TABLE_TOP = TABLE_HEIGHT + TABLE_SIZE[2] / 2   # actual surface height everything sits on
 
WALL_Y = 0.75               # window wall sits just behind the table's back edge
SILL_HEIGHT = 1.15           # window sill height, within arm 1's reach
SILL_DEPTH = 0.30            # deep enough for a pot to sit clear of the wall AND stay on the sill
WALL_MARGIN = 0.03           # minimum gap kept between any pot and the wall face
 
# Both arms bolted to the table SURFACE, spaced along its length
ARM1_BASE = SE3(0.25, 0.20, TABLE_TOP)   # TX2-60, sill-pick side
ARM2_BASE = SE3(1.05, 0.20, TABLE_TOP)   # reBot, pouring side
 
WINDOW_SILL_POSITION = SE3(0.25, WALL_Y, SILL_HEIGHT)  # plant's sill slot, above arm 1
WALL_FRONT_FACE_Y = WALL_Y + 0.01     # the wall's near face (pose offset 0.02 back, half-thickness 0.01)
SILL_FRONT_EDGE_Y = WALL_Y - SILL_DEPTH / 2   # the sill's own outer edge, facing the room
# Pot sits midway between the sill's front edge and the wall (minus its margin), so it's
# supported by the sill AND clear of the wall — valid for pots up to ~120 mm diameter at this depth.
SILL_FRONT_Y = (SILL_FRONT_EDGE_Y + (WALL_FRONT_FACE_Y - WALL_MARGIN)) / 2
PLATFORM_POSITION = SE3(0.65, 0.30, TABLE_TOP)     # midpoint on the table surface
CAN_HOLDER_POSITION = SE3(1.05, 0.45, TABLE_TOP)   # fixed spot near arm 2
 
 
# ---------------------------------------------------------------------
# Scene objects (placeholder boxes/cylinders — swap for real meshes later
# if you want it to look polished; geometry/position is what matters for
# collision checking and trajectory planning right now)
# ---------------------------------------------------------------------

def build_table():
    return geometry.Cuboid(
        scale=TABLE_SIZE,
        pose=TABLE_CENTRE,
        color=[0.75, 0.6, 0.45, 1],
    )

def build_wall():
    """A large flat wall panel in the x-z plane, behind the table, so the
    window sill has something to sit against rather than floating."""
    return geometry.Cuboid(
        scale=[1.6, 0.02, 1.8],
        pose=SE3(0.7, WALL_Y + 0.02, 0.9),
        color=[0.85, 0.85, 0.82, 1],
    )

def build_window_frame():
    """A rectangular window frame on the wall, above the sill, made from
    four thin bars. Returns a list of shapes — add each one to env."""
    bar_colour = [0.45, 0.32, 0.20, 1]
    cx, cz = 0.25, SILL_HEIGHT + 0.45   # window centred above the sill
    w, h, bar = 0.55, 0.65, 0.04
    y = WALL_Y + 0.005  # sits just in front of the wall face
    return [
        geometry.Cuboid(scale=[w, 0.02, bar], pose=SE3(cx, y, cz + h / 2), color=bar_colour),  # top
        geometry.Cuboid(scale=[w, 0.02, bar], pose=SE3(cx, y, cz - h / 2), color=bar_colour),  # bottom
        geometry.Cuboid(scale=[bar, 0.02, h], pose=SE3(cx - w / 2, y, cz), color=bar_colour),  # left
        geometry.Cuboid(scale=[bar, 0.02, h], pose=SE3(cx + w / 2, y, cz), color=bar_colour),  # right
    ]

def build_window_sill():
    """A window sill ledge mounted flush against the wall, above arm 1."""
    return geometry.Cuboid(
        scale=[0.6, SILL_DEPTH, 0.03],
        pose=WINDOW_SILL_POSITION,
        color=[0.9, 0.9, 0.88, 1],  # painted white sill, distinct from the wooden table
    )
def build_platform():
    return geometry.Cylinder(
        radius=0.08, length=0.02,
        pose=PLATFORM_POSITION * SE3(0, 0, 0.01),
        color=[0.6, 0.6, 0.6, 1],
    )

def build_pot(position: SE3, diameter_mm: float, height_mm: float, color):
    """A simple cylinder standing in for a potted plant."""
    return geometry.Cylinder(
        radius=(diameter_mm / 1000) / 2,
        length=height_mm / 1000,
        pose=position * SE3(0, 0, (height_mm / 1000) / 2),
        color=color,
    )

def build_can_holder():
    return geometry.Cylinder(
        radius=0.04, length=0.15,
        pose=CAN_HOLDER_POSITION * SE3(0, 0, 0.075),
        color=[0.2, 0.5, 0.8, 1],
    )


# ---------------------------------------------------------------------
# Placeholder for the Stäubli TX2-60
# ---------------------------------------------------------------------

def build_arm1_placeholder():
    # TODO: replace with `from models.staubli_tx2_60 import StaubliTX260`
    # once the TX2-60 model is merged.
    arm = rtb.models.DH.Puma560()
    arm.base = ARM1_BASE
    return arm


# ---------------------------------------------------------------------
# Build and launch the scene
# ---------------------------------------------------------------------

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    arm1 = build_arm1_placeholder()
    arm2 = ReBotB601DM()
    arm2.base = ARM2_BASE

    attach_stick_geometry(arm1)

    env.add(build_table())       
    env.add(arm1)
    env.add(arm2)

    env.add(build_wall())
    for bar in build_window_frame():
        env.add(bar)
    env.add(build_window_sill())
    env.add(build_platform())
    env.add(build_can_holder())


    # One pot on the sill as a starting point, matching PotProfile sizes
    pot_position = SE3(WINDOW_SILL_POSITION.t[0], SILL_FRONT_Y, SILL_HEIGHT)
    env.add(build_pot(pot_position * SE3(0, 0, 0.015), diameter_mm=120,
                       height_mm=140, color=[0.3, 0.6, 0.3, 1]))

    arm1.q = arm1.qz if hasattr(arm1, "qz") else np.zeros(arm1.n)
    arm2.q = np.zeros(6)
    env.step()

    print("Scene loaded. Check the Swift tab in your browser.")
    print(f"Table top surface at z={TABLE_TOP:.3f} m, both arm bases mounted on it")
    print(f"Arm 1 (TX2-60, placeholder DH model) base: {ARM1_BASE.t}")
    print(f"Arm 2 (reBot B601-DM) base: {ARM2_BASE.t}")
    print(f"Window sill: {WINDOW_SILL_POSITION.t}")
    print(f"Platform: {PLATFORM_POSITION.t}, Can holder: {CAN_HOLDER_POSITION.t}")

    env.hold()


if __name__ == "__main__":
    main()