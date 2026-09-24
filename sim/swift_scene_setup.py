"""
ShelfCare Swift scene setup.

Two robots, a shelf, a platform, and a wateruing-can holder,
each at a fixed (predetermined) location. This is scene setup only
"""

import swift
import spatialgeometry as geometry
from spatialmath import SE3
from math import pi
import numpy as np
import roboticstoolbox as rtb

from rebot_b601_dm import ReBotB601DM  # your model, from models/rebot_b601_dm.py


# ---------------------------------------------------------------------
# Office layout: one large table, both arm bases mounted on its surface,
# a shelf mounted on the wall behind/above the table.
# TODO: adjust these once you've measured your actual desk/wall layout
# ---------------------------------------------------------------------

TABLE_HEIGHT = 0.75        # standard desk height
TABLE_SIZE = [1.4, 0.7, 0.03]   # length (x) x depth (y) x thickness
TABLE_CENTRE = SE3(0.7, 0.35, TABLE_HEIGHT)

WALL_Y = 0.75               # window wall sits just behind the table's back edge
SILL_HEIGHT = 1.15           # window sill height, within arm 1's reach

# Both arms bolted to the table surface, spaced along its length
ARM1_BASE = SE3(0.25, 0.20, TABLE_HEIGHT)   # TX2-60, sill-pick side
ARM2_BASE = SE3(1.05, 0.20, TABLE_HEIGHT)   # reBot, pouring side

WINDOW_SILL_POSITION = SE3(0.25, WALL_Y, SILL_HEIGHT)  # plant's sill slot, above arm 1
PLATFORM_POSITION = SE3(0.65, 0.30, TABLE_HEIGHT)     # midpoint on the table, both arms can reach
CAN_HOLDER_POSITION = SE3(1.05, 0.45, TABLE_HEIGHT)   # fixed spot near arm 2


# Scene objects


def build_table():
    return geometry.Cuboid(
        scale=TABLE_SIZE,
        pose=TABLE_CENTRE,
        color=[0.75, 0.6, 0.45, 1],
    )

def build_window_sill():
    """A window sill ledge behind the table, above arm 1, where the plant lives."""
    return geometry.Cuboid(
        scale=[0.35, 0.06, 0.02],
        pose=WINDOW_SILL_POSITION,
        color=[0.9, 0.9, 0.88, 1],  # painted white sill, distinct from the wooden table
    )

def build_platform():
    return geometry.Cylinder(
        radius=0.08, length=0.02,
        pose=PLATFORM_POSITION * SE3(0, 0, -0.01),
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
# Placeholder TX2-60 
# ---------------------------------------------------------------------

def build_placeholder_arm1():
    # TODO: replace with `from staubli_tx2_60 import StaubliTX260`
    arm = rtb.models.DH.Puma560()
    arm.base = ARM1_BASE
    return arm


# ---------------------------------------------------------------------
# Build and launch the scene
# ---------------------------------------------------------------------

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    arm1 = build_placeholder_arm1()
    arm2 = ReBotB601DM()
    arm2.base = ARM2_BASE

    env.add(build_table())        # add furniture first, arms mount on top of it
    env.add(arm1)
    env.add(arm2)

    env.add(build_window_sill())
    env.add(build_platform())
    env.add(build_can_holder())

    # One pot on the sill as a starting point, matching PotProfile sizes
    # from shelfcare_sequence.py — swap in the real registry values here.
    env.add(build_pot(WINDOW_SILL_POSITION * SE3(0, 0, 0.02), diameter_mm=120,
                       height_mm=140, color=[0.3, 0.6, 0.3, 1]))

    arm1.q = arm1.qz if hasattr(arm1, "qz") else np.zeros(arm1.n)
    arm2.q = np.zeros(6)
    env.step()

    print("Scene loaded. Check the Swift tab in your browser.")
    print(f"Table top at z={TABLE_HEIGHT} m, both arm bases mounted on it")
    print(f"Arm 1 (TX2-60 placeholder) base: {ARM1_BASE.t}")
    print(f"Arm 2 (reBot) base: {ARM2_BASE.t}")
    print(f"Window sill: {WINDOW_SILL_POSITION.t}")
    print(f"Platform: {PLATFORM_POSITION.t}, Can holder: {CAN_HOLDER_POSITION.t}")

    env.hold()


if __name__ == "__main__":
    main()
