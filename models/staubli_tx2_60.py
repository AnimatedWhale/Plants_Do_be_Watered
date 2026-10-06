"""
Stäubli TX2-60 kinematic model for the Robotics Toolbox for Python.

Three classes describe the same arm:

    StaubliTX260DH   standard (distal) DH model, rtb.DHRobot. Use this for the
                     DH table in the report, fkine / ikine / jacobians, teach().
    StaubliTX260MDH  the same arm in modified (Craig) DH, in case your course
                     uses that convention.
    StaubliTX260     link-by-link ERobot with simple primitive geometry so it
                     renders in Swift (a DHRobot has no meshes, so Swift draws
                     nothing for it). This is the one the scene uses.

All three follow the controller's joint convention: q = 0 is the arm standing
straight up, so joint angles here match the teach pendant 1:1.

Run this file directly to print the DH table and check that all three
models give the same forward kinematics.

Sources
    [1] ROS-Industrial, staubli_experimental, staubli_tx2_60_support/urdf/
        tx2_60_macro.xacro: joint origins, joint limits and max joint speeds.
        https://github.com/ros-industrial/staubli_experimental
    [2] Stäubli TX2-60 product page (reach 670 mm).
        https://www.staubli.com/global/en/robotics/products/industrial-robots/6-axis/tx2-60.html
"""

from math import pi

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from roboticstoolbox import ET, Link, RevoluteDH, RevoluteMDH
from spatialmath import SE3


# ---------------------------------------------------------------------
# Geometry (metres), from [1]. 290 + 310 + 70 = 670 mm reach, matches [2].
# ---------------------------------------------------------------------

D1 = 0.375   # floor of the base to the shoulder (J2) axis
A2 = 0.290   # shoulder (J2) to elbow (J3): upper arm
D3 = 0.020   # sideways offset between J2 and J3, measured along the J2 axis
D4 = 0.310   # elbow (J3) to wrist centre (J5): forearm
D6 = 0.070   # wrist centre (J5) to tool flange

# Joint limits (rad), from [1]
QLIM = np.deg2rad([
    [-180.0, 180.0],
    [-127.5, 127.5],
    [-152.5, 152.5],
    [-270.0, 270.0],
    [-121.0, 132.5],
    [-270.0, 270.0],
])

# Max joint speeds (rad/s), from [1]
QD_MAX = np.deg2rad([435, 385, 500, 995, 1065, 1445])

# Flange -> TCP (the point midway between the gripper fingers). The fingers
# have to reach past the centre of the widest pot (160 mm), and open wide
# enough to straddle it.
# TODO: measure your gripper and update these.
GRIPPER_LENGTH = 0.15
FINGER_OFFSET = 0.085      # each finger's distance from the TCP along tool y (170 mm opening)
TOOL = SE3(0, 0, GRIPPER_LENGTH)

# Named configurations
QZ = np.zeros(6)                       # arm straight up
QR = np.deg2rad([0, 0, 90, 0, 0, 0])   # forearm horizontal, tool pointing +x, tool x-axis down


# ---------------------------------------------------------------------
# DH models
# ---------------------------------------------------------------------

class StaubliTX260DH(rtb.DHRobot):
    """Standard DH:  T_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)."""

    def __init__(self, tool=TOOL):
        links = [
            RevoluteDH(d=D1, a=0.0, alpha=-pi / 2, offset=0.0,    qlim=QLIM[0]),
            RevoluteDH(d=0.0, a=A2, alpha=0.0,     offset=-pi / 2, qlim=QLIM[1]),
            RevoluteDH(d=D3, a=0.0, alpha=pi / 2,  offset=pi / 2,  qlim=QLIM[2]),
            RevoluteDH(d=D4, a=0.0, alpha=-pi / 2, offset=0.0,    qlim=QLIM[3]),
            RevoluteDH(d=0.0, a=0.0, alpha=pi / 2, offset=0.0,    qlim=QLIM[4]),
            RevoluteDH(d=D6, a=0.0, alpha=0.0,     offset=0.0,    qlim=QLIM[5]),
        ]
        super().__init__(links, name="TX2-60 (DH)", manufacturer="Stäubli", tool=tool)
        self.addconfiguration("qz", QZ)
        self.addconfiguration("qr", QR)


class StaubliTX260MDH(rtb.DHRobot):
    """Modified DH:  T_i = Rx(alpha_{i-1}) Tx(a_{i-1}) Rz(theta_i) Tz(d_i).
    Frame 6 sits at the wrist centre, so the 70 mm flange goes into the tool."""

    def __init__(self, gripper_length=GRIPPER_LENGTH):
        links = [
            RevoluteMDH(d=D1, a=0.0, alpha=0.0,     offset=0.0,    qlim=QLIM[0]),
            RevoluteMDH(d=0.0, a=0.0, alpha=-pi / 2, offset=-pi / 2, qlim=QLIM[1]),
            RevoluteMDH(d=D3, a=A2, alpha=0.0,     offset=pi / 2,  qlim=QLIM[2]),
            RevoluteMDH(d=D4, a=0.0, alpha=pi / 2,  offset=0.0,    qlim=QLIM[3]),
            RevoluteMDH(d=0.0, a=0.0, alpha=-pi / 2, offset=0.0,    qlim=QLIM[4]),
            RevoluteMDH(d=0.0, a=0.0, alpha=pi / 2,  offset=0.0,    qlim=QLIM[5]),
        ]
        super().__init__(links, name="TX2-60 (MDH)", manufacturer="Stäubli",
                         tool=SE3(0, 0, D6 + gripper_length))
        self.addconfiguration("qz", QZ)
        self.addconfiguration("qr", QR)


# ---------------------------------------------------------------------
# Swift-renderable model (same kinematics, built joint by joint as in [1])
# ---------------------------------------------------------------------

_BODY = [0.92, 0.92, 0.90, 1.0]
_ACCENT = [0.95, 0.70, 0.10, 1.0]
_DARK = [0.25, 0.25, 0.28, 1.0]


def _y_hub(radius, length, color):
    """Cylinder lying along the local y axis: the housing of a J2/J3/J5 joint."""
    return sg.Cylinder(radius=radius, length=length, pose=SE3.Rx(pi / 2), color=color)


def _z_rod(radius, z0, z1, color):
    """Cylinder along the local z axis from z0 to z1."""
    return sg.Cylinder(radius=radius, length=z1 - z0, pose=SE3(0, 0, (z0 + z1) / 2), color=color)


def _gripper_geometry(length=GRIPPER_LENGTH):
    """Flange, palm and two fingers. Fingers close along the tool y axis and
    their mid-point sits at the TCP (`length` out from the flange)."""
    finger = length - 0.01
    return [
        _z_rod(0.032, -0.01, 0.01, _DARK),
        sg.Cuboid(scale=[0.05, 2 * FINGER_OFFSET + 0.02, 0.03], pose=SE3(0, 0, 0.015), color=_DARK),
        sg.Cuboid(scale=[0.03, 0.012, finger], pose=SE3(0, FINGER_OFFSET, 0.03 + finger / 2), color=_DARK),
        sg.Cuboid(scale=[0.03, 0.012, finger], pose=SE3(0, -FINGER_OFFSET, 0.03 + finger / 2), color=_DARK),
    ]


class StaubliTX260(rtb.Robot):
    """TX2-60 with primitive geometry for Swift. Joint origins follow [1]:
    J1 about z at 375 mm, J2 about y, J3 about y at (0, 20, 290) mm from J2,
    J4 about z, J5 about y 310 mm up the forearm, flange 70 mm past J5."""

    def __init__(self, tool=TOOL):
        l1 = Link(ET.tz(D1) * ET.Rz(), name="link1", qlim=QLIM[0], geometry=[
            _z_rod(0.10, -D1, -D1 + 0.20, _DARK),     # base plinth
            _z_rod(0.08, -D1 + 0.20, 0.0, _BODY),     # turret
        ])
        l2 = Link(ET.Ry(), name="link2", parent=l1, qlim=QLIM[1], geometry=[
            _y_hub(0.075, 0.16, _ACCENT),
            _z_rod(0.05, 0.0, A2, _BODY),             # upper arm
        ])
        l3 = Link(ET.ty(D3) * ET.tz(A2) * ET.Ry(), name="link3", parent=l2, qlim=QLIM[2], geometry=[
            _y_hub(0.06, 0.13, _ACCENT),
            _z_rod(0.042, 0.0, D4 - 0.07, _BODY),     # forearm
        ])
        l4 = Link(ET.Rz(), name="link4", parent=l3, qlim=QLIM[3], geometry=[
            _z_rod(0.038, D4 - 0.07, D4 - 0.02, _BODY),  # wrist housing, rolls with J4
        ])
        l5 = Link(ET.tz(D4) * ET.Ry(), name="link5", parent=l4, qlim=QLIM[4], geometry=[
            _y_hub(0.04, 0.09, _ACCENT),
            _z_rod(0.03, 0.0, D6, _BODY),
        ])
        l6 = Link(ET.tz(D6) * ET.Rz(), name="link6", parent=l5, qlim=QLIM[5],
                  geometry=_gripper_geometry())

        super().__init__([l1, l2, l3, l4, l5, l6], name="TX2-60", manufacturer="Stäubli", tool=tool)
        self.qd_max = QD_MAX
        self.qz, self.qr = QZ, QR   # rtb.Robot's addconfiguration doesn't set these attributes
        self.addconfiguration("qz", QZ)
        self.addconfiguration("qr", QR)


# ---------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------

def check_models(n=500, seed=0):
    """Largest FK difference (m / unitless) between the three models over
    random joint angles. Should be ~1e-15."""
    rng = np.random.default_rng(seed)
    dh, mdh, er = StaubliTX260DH(), StaubliTX260MDH(), StaubliTX260()
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(QLIM[:, 0], QLIM[:, 1])
        T = er.fkine(q).A
        worst = max(worst, np.abs(dh.fkine(q).A - T).max(), np.abs(mdh.fkine(q).A - T).max())
    return worst


if __name__ == "__main__":
    dh = StaubliTX260DH()
    print(dh)
    print("TCP at q = 0 :", np.round(dh.fkine(QZ).t, 4), f" expected [0, 0.02, {1.045 + GRIPPER_LENGTH:.3f}]")
    print("TCP at q = qr:", np.round(dh.fkine(QR).t, 4), f" expected [{0.38 + GRIPPER_LENGTH:.2f}, 0.02, 0.665]")
    worst = check_models()
    print(f"DH vs MDH vs Swift model, worst FK difference: {worst:.2e}",
          "OK" if worst < 1e-9 else "MISMATCH")
