"""
ShelfCare watering sequence.

Arm 1 (Stäubli TX2-60) carries each dry plant from the shelf to the platform
and back. Arm 2 (reBot B601-DM) picks up the watering can and pours.

Every move is checked for collisions before it starts.

Run from the repo root:
    python Jayden-Bot/sim/Selfcare_sequence.py                   # open Swift
    python Jayden-Bot/sim/Selfcare_sequence.py --moisture 10     # water every plant
    python Jayden-Bot/sim/Selfcare_sequence.py --speed 1         # real-time playback
    python Jayden-Bot/sim/Selfcare_sequence.py --collision-test  # drop an object in the way
"""

import argparse
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from math import atan2

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from swift_scene_setup import (ARM1_PARK, CAN_HOLDER_POSITION, CAN_HOLDER_TOP, PLATFORM_POSITION,
                               PLATFORM_TOP, REBOT_FINGER_OPEN, SHELF_CENTRE, SHELF_HEIGHT, SHELF_SIZE, TABLE_CENTRE,
                               TABLE_HEIGHT, TABLE_SIZE, DroppedObject, Pot, WateringCanDuck, build_scene)
from collision import Box, VCylinder, Spheres, capsule, cylinder, first_collision, rebot_body, tx2_60_body
from models.staubli_tx2_60 import FINGER_OFFSET, GRIPPER_LENGTH, StaubliTX260DH


# ---------------------------------------------------------------------
# Pot geometry registry
# ---------------------------------------------------------------------

@dataclass
class PotProfile:
    pot_id: str
    diameter_mm: float
    height_mm: float
    shelf_position: tuple
    moisture_threshold: float
    water_volume_ml: float

    @property
    def grasp_offset_z(self) -> float:
        return -0.6 * (self.height_mm / 1000)

    @property
    def pour_clearance_z(self) -> float:
        return (self.height_mm / 1000) + 0.04


POT_REGISTRY = {
    "pot_small": PotProfile(
        pot_id="pot_small", diameter_mm=80, height_mm=90,
        shelf_position=(0.02, 0.66),
        moisture_threshold=30.0, water_volume_ml=100,
    ),
    "pot_medium": PotProfile(
        pot_id="pot_medium", diameter_mm=120, height_mm=140,
        shelf_position=(0.25, 0.66),
        moisture_threshold=35.0, water_volume_ml=200,
    ),
    "pot_large": PotProfile(
        pot_id="pot_large", diameter_mm=160, height_mm=190,
        shelf_position=(0.48, 0.66),
        moisture_threshold=40.0, water_volume_ml=350,
    ),
}


# ---------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------

DT = 0.05
DEFAULT_SPEED = 5
# Every move is timed from how far it goes, easing in and out, so nothing
# jumps or crawls. Speeds are real time (before the --speed playback factor).
TCP_SPEED = 0.20                # m/s, top gripper speed on straight-line and arc moves
TCP_TURN_SPEED = np.deg2rad(60) # rad/s, top gripper turning speed
LOWER_SPEED = 0.08              # m/s, bringing a pot down from its hover over the platform
SET_DOWN_SPEED = 0.04           # m/s, the last few cm onto a surface, and lifting off one
JOINT_SPEED_FRACTION = 0.2      # free joint moves use at most this share of the TX2-60's rated joint speeds...
JOINT_MOVE_TCP_SPEED = 0.40     # m/s ...and the gripper still stays under this on them
REBOT_JOINT_SPEED = 1.0         # rad/s per joint for the reBot, whose model has no rated speeds
MIN_STEPS = 10
GRIP_TICKS = 10
ARM1_BACKOFF = 0.15
ARM2_BACKOFF = 0.08
ARM1_PITCH = np.deg2rad(45)     # grip angled down: keeps the wrist bent (away from its singularity) at the shelf
ARM1_MIN_GRIP_HEIGHT = 0.05      # m above the pot base, so angled-down fingertips clear the shelf
ARM1_GRIP_BELOW_RIM = 0.05       # m, grip at most this far under the rim so the wrist clears a tall pot
ARM2_GRIP_DEPTH = 0.02          # how far the fingertips reach past the centre of the can's handle bar
LIFT = 0.04
SHELF_CLEARANCE = 0.30
PLATFORM_HOVER = 0.12            # m, carried pot comes round this far above the platform, then lowers
CARRY_DROP = (0.35, 1.0)          # height changes only in the 2nd half of the swing to the platform...
CARRY_RISE = (0.0, 0.5)          # ...and the 1st half of the swing back, so the pot is high past the shelf
PLATFORM_APPROACH_OFFSETS = [0, 15, -15, 30, -30, 45, -45, 60, -60]   # deg off head-on; head-on keeps the wrist untwisted
MAX_CARRY_WRIST_ROLL = 250.0     # deg of J4 + J6 turning allowed while carrying a pot to the platform (one smooth roll)
SHELF_RETREAT = 0.15             # m, how far a lifted pot is pulled back (along the grip axis) before swinging round
CARRY_LIFT = 0.10
CARRY_MAX_TILT = np.deg2rad(10)  # the full can may lean this much while it is carried
CARRY_HUMPS = (0.0, 0.08, 0.15)  # extra height tried in the middle of a level swing
POUR_ANGLE = np.deg2rad(80)     # how far the can tips to pour
POUR_STEPS = 60                 # waypoints in the pour tilt
POUR_RIM_INSET = 0.015          # spout tip sits this far inside the pot rim
POUR_FLOOR_MARGIN = 0.02        # gap kept between the can and the platform
POUR_APPROACH_LIFT = 0.10       # the can arrives this high above the pour start
POUR_RATE_ML_S = 50.0
MAX_JOINT_STEP = 0.3
ZONE_AUTO_RESUME = True         # True: carry on when the light curtain clears. False: needs Acknowledge + Resume
MIN_UPRIGHT_COS = 0.0           # the can must never tip past level (0 = level)

# Manual (teach) mode. ISO 10218-1 caps the TCP at 250 mm/s in reduced-speed
# manual mode, and manual moves play in real time (no --speed factor).
MANUAL_JOINT_SPEED_FRACTION = 0.10   # share of the TX2-60's rated joint speeds when jogging
MANUAL_TCP_SPEED = 0.25              # m/s, cap on the gripper during a manual joint move
MANUAL_WORK_REACH = 0.15             # m, the object nearest the gripper within this may be reached into
GRAB_TOLERANCE = 0.02                # m, how far the TCP may be off an object and still grab it
ARM1_FINGER_CLOSED = 0.01            # TX2-60 finger offset with nothing in the gripper
REBOT_FINGER_CLOSED = 0.003          # reBot finger travel with nothing in the gripper

DOWN = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------
# Simulated moisture sensor
# ---------------------------------------------------------------------

def read_moisture(pot_id: str) -> float:
    return random.uniform(10, 60)


# ---------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------

class PlanningError(RuntimeError):
    pass


def horizontal_unit(v):
    v = np.array([v[0], v[1], 0.0])
    return v / np.linalg.norm(v)


def side_grasp(point, approach, pitch=0.0) -> SE3:
    """Grasp pose at `point`. Tool z points along `approach`, tilted down by
    `pitch`. The fingers close sideways."""
    h = horizontal_unit(approach)
    a = np.cos(pitch) * h + np.sin(pitch) * DOWN
    x = np.cos(pitch) * DOWN - np.sin(pitch) * h
    return SE3(point) * SE3.OA(np.cross(a, x), a)


def can_stays_upright(R_carry, R_can_in_tool, tilt_axis_fn, sign, max_angle, steps=9) -> bool:
    """True if the can never tips past level during the pour."""
    for deg in np.linspace(0, np.rad2deg(max_angle), steps):
        R = R_carry @ tilt_axis_fn(sign * np.deg2rad(deg)).R
        up = (R @ R_can_in_tool)[:, 2]
        if up[2] < MIN_UPRIGHT_COS:
            return False
    return True


def transform_point(T: SE3, p):
    return (T.A @ np.append(p, 1.0))[:3]


def _ik_goal(arm, T_goal: SE3) -> SE3:
    if not hasattr(arm, "_ik_in_base_frame"):
        T_here = arm.fkine(arm.q)
        sol = arm.ikine_LM(T_here, q0=arm.q, tol=1e-10, slimit=1)
        arm._ik_in_base_frame = not (sol.success and np.abs(arm.fkine(sol.q).A - T_here.A).max() < 1e-6)
    return arm.base.inv() * T_goal if arm._ik_in_base_frame else T_goal


def solve_ik(arm, T_goal: SE3, q0, slimit=100):
    sol = arm.ikine_LM(_ik_goal(arm, T_goal), q0=q0, joint_limits=True, tol=1e-10, slimit=slimit)
    if sol.success and np.abs(arm.fkine(sol.q).A - T_goal.A).max() < 1e-3:
        return _nearest(arm, sol.q, q0)
    raise PlanningError(f"{arm.name}: no IK solution for TCP at {np.round(T_goal.t, 3)}")


def _nearest(arm, q, q_ref):
    """The joint vector for the same pose as q that is nearest q_ref, for arms
    that have more than one (the TX2-60's wrist can flip, and J1/J4/J6 can go
    a full turn either way). Without it the IK picks freely and the wrist
    spins extra half and full turns between poses."""
    return arm.closest_equivalent(q, q_ref) if hasattr(arm, "closest_equivalent") else q


# rtb's jtraj / ctraj ease in and out; at their fastest they run 1.875x the average speed
PEAK = 1.875


def _ticks(duration):
    """Waypoints for a smooth move lasting `duration` seconds."""
    return max(MIN_STEPS, int(np.ceil(duration / DT)) + 1)


def joint_move(q_from, q_to, vmax):
    """Joint-space move timed so no joint goes faster than vmax (rad/s, per joint)."""
    dq = np.abs(np.asarray(q_to, float) - np.asarray(q_from, float))
    steps = _ticks(PEAK * float((dq / vmax).max()))
    return list(rtb.jtraj(q_from, q_to, steps).q)


def cartesian_move(arm, q_from, T_to: SE3, speed=TCP_SPEED):
    """Straight-line move, timed so the gripper tops out at `speed`."""
    T_from = arm.fkine(q_from)
    dist = np.linalg.norm(T_to.t - T_from.t)
    turn = abs((T_from.inv() * T_to).angvec()[0])
    steps = _ticks(PEAK * max(dist / speed, turn / TCP_TURN_SPEED))
    return follow_poses(arm, q_from, rtb.ctraj(T_from, T_to, steps))


def arc_move(arm, q_from, T_to: SE3, height_change=(0.0, 1.0), speed=TCP_SPEED):
    """Swing round the arm's base axis. The height only changes during the
    `height_change` part of the swing (fractions of it), so a carried pot can
    stay high while it passes the shelf plants and drop only once clear. The
    tool turns only about the vertical, so a carried pot stays upright: it
    swings round with the arm, and any extra turn needed to reach T_to's
    heading is made along with the height change.

    The swing is about the arm's own plane, which for the TX2-60 sits
    `plane_offset` (its 20 mm D3) to the side of the base axis. Keeping the
    gripper in that plane the whole way means J4 and J6 can stay at zero and
    only J5 bends the wrist (up and down), even where it passes straight."""
    T_from = arm.fkine(q_from)
    c = arm.base.t
    side = getattr(arm, "plane_offset", 0.0)
    plane = arm.arm_heading if hasattr(arm, "arm_heading") else (lambda p: atan2(p[1] - c[1], p[0] - c[0]))

    def reach(p):   # distance out along the arm's plane
        return np.sqrt(max(np.hypot(p[0] - c[0], p[1] - c[1]) ** 2 - side ** 2, 0.0))

    a0 = plane(T_from.t)
    da = (plane(T_to.t) - a0 + np.pi) % (2 * np.pi) - np.pi
    r0, r1 = reach(T_from.t), reach(T_to.t)
    z0, z1 = T_from.t[2], T_to.t[2]
    heading = lambda T: atan2(T.R[1, 2], T.R[0, 2])
    turn = (heading(T_to) - heading(T_from) + np.pi) % (2 * np.pi) - np.pi
    h0, h1 = height_change

    def path(steps):
        poses = []
        for t in np.linspace(0.0, 1.0, steps):
            s = (1 - np.cos(np.pi * t)) / 2
            u = np.clip((s - h0) / (h1 - h0), 0.0, 1.0)
            sz = (1 - np.cos(np.pi * u)) / 2
            a, r, z = a0 + s * da, r0 + s * (r1 - r0), z0 + sz * (z1 - z0)
            # follow the swing; any extra turn happens with the height change
            R = SE3.Rz(s * da + sz * (turn - da)).R @ T_from.R
            out, across = np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])
            poses.append(SE3.Rt(R, c + r * out + side * across + [0.0, 0.0, z - c[2]]))
        return poses

    # Time it so the fastest stretch of the swing runs at `speed`
    trial = path(50)
    move = max(np.linalg.norm(b.t - a.t) for a, b in zip(trial, trial[1:]))
    spin = max(abs((a.inv() * b).angvec()[0]) for a, b in zip(trial, trial[1:]))
    steps = max(MIN_STEPS, int(np.ceil(49 * max(move / (speed * DT), spin / (TCP_TURN_SPEED * DT)))) + 1)
    return follow_poses(arm, q_from, path(steps))


def upright_move(arm, q_from, T_to: SE3, hump=0.0, speed=TCP_SPEED):
    """Swing around the arm's base to T_to, turning only about the vertical,
    so whatever is held stays level. `hump` lifts the middle of the swing.
    Timed so the gripper tops out at `speed`."""
    T_from = arm.fkine(q_from)
    turn = T_to.R @ T_from.R.T
    yaw = atan2(turn[1, 0], turn[0, 0])
    c = arm.base.t
    p0, p1 = T_from.t - c, T_to.t - c
    r0, r1 = np.hypot(*p0[:2]), np.hypot(*p1[:2])
    a0 = atan2(p0[1], p0[0])
    da = (atan2(p1[1], p1[0]) - a0 + np.pi) % (2 * np.pi) - np.pi

    def path(steps):
        poses = []
        for t in np.linspace(0.0, 1.0, steps):
            s = (1 - np.cos(np.pi * t)) / 2
            a, r, z = a0 + s * da, r0 + s * (r1 - r0), p0[2] + s * (p1[2] - p0[2]) + hump * np.sin(np.pi * s)
            poses.append(SE3.Rt(SE3.Rz(s * yaw).R @ T_from.R, c + [r * np.cos(a), r * np.sin(a), z], check=False))
        return poses

    trial = path(50)
    length = sum(np.linalg.norm(b.t - a.t) for a, b in zip(trial, trial[1:]))
    return follow_poses(arm, q_from, path(_ticks(PEAK * max(length / speed, abs(yaw) / TCP_TURN_SPEED))))


def follow_poses(arm, q_from, poses):
    max_step = arm.qd_max * DT if hasattr(arm, "qd_max") else np.full(arm.n, MAX_JOINT_STEP)
    qs, q = [], q_from
    for T in poses:
        q_next = solve_ik(arm, T, q)
        over = np.abs(q_next - q) / max_step
        if over.max() > 1.0:
            j = over.argmax()
            raise PlanningError(f"{arm.name}: configuration flip on a Cartesian move "
                                f"(J{j + 1} jumps {np.rad2deg(abs(q_next[j] - q[j])):.0f} deg in one tick)")
        qs.append(q_next)
        q = q_next
    return qs


def back(T_grasp: SE3, distance) -> SE3:
    return T_grasp * SE3(0, 0, -distance)


def up(T: SE3, height) -> SE3:
    return SE3(0, 0, height) * T


# ---------------------------------------------------------------------
# Sequence state machine
# ---------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    CHECK_MOISTURE = auto()
    ARM1_TO_SHELF = auto()
    ARM1_APPROACH_POT = auto()
    ARM1_GRASP = auto()
    ARM1_LIFT_FROM_SHELF = auto()
    ARM1_TO_PLATFORM = auto()
    ARM1_LOWER_ONTO_PLATFORM = auto()
    ARM1_RELEASE = auto()
    ARM1_RETREAT = auto()
    ARM2_GET_CAN = auto()
    ARM2_APPROACH_CAN = auto()
    ARM2_GRASP_CAN = auto()
    ARM2_LIFT_CAN = auto()
    ARM2_TO_POUR_POSITION = auto()
    ARM2_POUR = auto()
    ARM2_UNTILT = auto()
    ARM2_RETURN_CAN = auto()
    ARM2_RELEASE_CAN = auto()
    ARM2_HOME = auto()
    ARM1_TO_PLATFORM_COLLECT = auto()
    ARM1_GRASP_PLANT = auto()
    ARM1_RETURN_TO_SHELF = auto()
    ARM1_RELEASE_ON_SHELF = auto()
    ARM1_HOME = auto()
    DONE = auto()
    FAULT = auto()


_ORDER = [s for s in State if s is not State.FAULT]


class ShelfCareSequence:
    def __init__(self, scene, force_moisture=None, speed=DEFAULT_SPEED):
        self.arm1, self.arm2, self.can, self.pots = scene.arm1, scene.arm2, scene.can, scene.pots
        self.force_moisture = force_moisture
        self.speed = max(1, int(speed))
        self._furniture = scene.obstacles
        self._dh1 = StaubliTX260DH()
        self._dh1.base = self.arm1.base

        self.state = State.IDLE
        self.estop_active = False
        self.fault_reason = ""
        self._resume_state = None

        self.current_pot = None
        self.keep_watching = False      # with the GUI: when no plant needs water, wait for one to dry out
        self.moisture = self._fresh_moisture()
        self._all_ok_reported = False

        self._queue = deque()
        self._hold = 0
        self._entered = False
        self._holder = None
        self._held = None
        self._T_tool_obj = None
        self._home2 = self.arm2.q.copy()

        self._scene = scene
        self._finger_goal = REBOT_FINGER_OPEN
        self._arm1_finger_goal = FINGER_OFFSET   # TX2-60 fingers: inner face distance from the TCP

        self.stop_latched = False               # stopped, needs Acknowledge
        self.waiting = False                    # reset, waiting for Resume
        self.zone_breached = False              # light curtain broken
        self._start = (self.arm1.q.copy(), self.arm2.q.copy(), self.can.T,
                       {pot_id: pot.T for pot_id, pot in self.pots.items()})
        self._transit_goal = None               # end of the current free joint move, if it is one
        self._hazard_checked = scene.hazard.version
        self._hazard_xyz = np.array([0.70, 0.35, 1.00])

        self.paused = False                     # Pause/Hold: a normal stop, no acknowledge needed
        self.override = 1.0                     # speed override, share of the programmed speed
        self._budget = 0.0                      # waypoints owed to playback (lets the override be fractional)
        self.manual = False                     # manual (teach) mode instead of the automatic sequence
        self.jog_arm = self.arm1                # the arm the manual controls move
        self.jog_enabled = False                # the enable switch: nothing jogs unless it is on
        self.jog_version = 0                    # bumped whenever the joint sliders should re-read the arm

    # --- safety -------------------------------------------------------

    def estop(self):
        """Stop both arms where they are, in any mode. It stays stopped until
        acknowledged, even if nothing was moving when it was pressed."""
        if self.estop_active:
            return
        self.estop_active = True
        if self.manual:
            self._queue.clear()             # a manual jog is just dropped, not resumed
        elif self.state not in (State.FAULT, State.DONE) and not self.waiting:
            self._resume_state = self.state
            self.state = State.FAULT
        if not self.stop_latched:
            self.fault_reason = "E-STOP pressed"
        self._log("E-STOP")

    def unsafe_zone(self, active: bool):
        """Light curtain signal. Motion pauses while it is broken."""
        if active == self.zone_breached:
            return
        self.zone_breached = active
        self._log("light curtain BROKEN - paused" if active else "light curtain clear")
        if active and not ZONE_AUTO_RESUME and self.state not in (State.FAULT, State.DONE) and not self.waiting:
            self._protective_stop("light curtain broken")
        if not active:
            self._hazard_checked = None     # re-check the path

    @property
    def signal(self):
        """Which signal tower lamp is lit: red stopped, amber paused / manual / idle, green running."""
        if self.stopped or self.state == State.FAULT:
            return "red"
        if self.zone_breached or self.paused or self.manual or self.waiting or self.state == State.DONE:
            return "amber"
        return "green"

    def acknowledge(self):
        """The operator has seen the stop and dealt with its cause. Nothing
        moves: this only makes resume() available again."""
        if self.estop_active or self.stop_latched:
            self.estop_active = False
            self.stop_latched = False
            self._log("acknowledged")
        else:
            self._log("nothing to acknowledge")

    def resume(self):
        """Start after a reset, carry on after Pause, or carry on with the
        interrupted task after a stop that has been acknowledged."""
        if self.estop_active or self.stop_latched:
            self._log("still stopped - press Acknowledge first")
        elif self.manual:
            self._log("in manual mode - switch to Auto to run the sequence")
        elif self.paused:
            self.paused = False
            self._hazard_checked = None     # look again at whatever is still queued before moving
            self._log("resumed")
        elif self.waiting:
            self.waiting = False
            self._log("started")
        elif self.state == State.FAULT and self._resume_state is not None:
            self.state, self._resume_state = self._resume_state, None
            self._hazard_checked = None
            self._log("resumed")
        elif self.state == State.FAULT:
            self._log("cannot carry on from this stop - press Reset to start over")

    def pause(self):
        """Hold: a normal (not safety) stop. The arm stops at its next
        waypoint and Resume carries on, no acknowledge needed."""
        if self.manual:
            self.jog_enabled = False
            self._queue.clear()
            self._log("manual jog stopped")
        elif self.waiting or self.state in (State.FAULT, State.DONE):
            self._log("nothing is running to pause")
        elif not self.paused:
            self.paused = True
            self._log("paused - press Start / Resume to carry on")

    def set_override(self, percent):
        self.override = float(np.clip(float(percent) / 100.0, 0.05, 1.0))

    def set_manual(self, manual):
        """Switch between Auto and Manual. Manual is only allowed while the
        sequence is not moving (paused, stopped, at the start or done).
        Going back to Auto resets the cell, since the arms are no longer
        where the sequence left them."""
        if manual == self.manual:
            return
        if manual:
            running = not (self.paused or self.waiting or self.state in (State.FAULT, State.DONE)
                           or self.estop_active)
            if running:
                self._log("press Pause (or E-STOP) before switching to Manual")
                return
            self._queue.clear()
            self._transit_goal = None
            self._resume_state = None
            self.paused = False
            self.manual = True
            self.jog_enabled = False
            self.jog_version += 1
            self._log("MANUAL mode - tick 'Enable jog' to move an arm")
        else:
            self.manual = False
            self.jog_enabled = False
            self.reset()
            self._log("AUTO mode - cell reset to the start, press Start / Resume to run")

    def reset(self):
        """Put everything back to the start and wait for Resume."""
        q1, q2, T_can, pot_poses = self._start
        self._queue.clear()
        self._hold = 0
        self._entered = False
        self._holder = self._held = self._T_tool_obj = None
        self._resume_state = None
        self._transit_goal = None
        self.arm1.q, self.arm2.q = q1.copy(), q2.copy()
        self.can.set_pose(T_can)
        for pot_id, T in pot_poses.items():
            self.pots[pot_id].set_pose(T)
        self._scene.hazard.stow()
        self._hazard_checked = self._scene.hazard.version
        self._finger_goal = self._scene.arm2_finger_opening = REBOT_FINGER_OPEN
        self._arm1_finger_goal = FINGER_OFFSET
        self.arm1.set_finger_offset(FINGER_OFFSET)
        self.current_pot = None
        self.moisture = self._fresh_moisture()
        self._all_ok_reported = False
        self.estop_active = self.stop_latched = False
        self.fault_reason = ""
        self.state = State.IDLE
        self.waiting = True
        self.paused = False
        self.jog_version += 1
        self._log("reset to the start")

    @property
    def status(self):
        """What the operator should see: the mode, the state, why it stopped, what to press."""
        if self.estop_active or self.stop_latched:
            return f"STOPPED - {self.fault_reason} - press Acknowledge"
        if self.manual:
            jog = "jogging" if self._queue else ("enabled" if self.jog_enabled else "tick Enable jog to move")
            curtain = " - PAUSED, light curtain broken" if self.zone_breached else ""
            return f"MANUAL - {self.jog_arm.name} - {jog}{curtain}"
        if self.waiting:
            return "AUTO - READY at the start - press Start / Resume to run"
        if self.state == State.DONE:
            return "AUTO - DONE - press Reset to go back to the start"
        if self.paused:
            return f"AUTO - PAUSED in {self.state.name} - press Start / Resume to carry on"
        if self.zone_breached and self.state != State.FAULT:
            return f"AUTO - PAUSED - light curtain broken during {self.state.name} - clear the zone to continue"
        if self.state == State.CHECK_MOISTURE and self.current_pot is None and self._all_ok_reported:
            return "AUTO - every plant is moist enough - watching the moisture levels"
        if self.state != State.FAULT:
            return f"AUTO - Running: {self.state.name}"
        if self._resume_state is not None:
            return f"AUTO - STOPPED, acknowledged - {self.fault_reason} - press Start / Resume to carry on"
        return f"AUTO - STOPPED - {self.fault_reason} - press Reset to start over"

    @property
    def stopped(self):
        """True while a latched stop (E-STOP or protective stop) is in force."""
        return self.estop_active or self.stop_latched

    def _fault(self, reason):
        self._log(f"FAULT: {reason}")
        self._queue.clear()
        self.fault_reason = f"{self.state.name}: {reason}"
        self._resume_state = None
        self.state = State.FAULT

    def _protective_stop(self, reason):
        """Stop where we are but keep the planned path, so Resume can carry on."""
        self._log(f"PROTECTIVE STOP: {reason}")
        self.fault_reason = f"{self.state.name}: {reason}"
        self._resume_state = self.state
        self.stop_latched = True
        self.state = State.FAULT

    # --- dropped object (collision test) ----------------------------------

    def drop_obstacle_ahead(self):
        """Drop the object on the moving arm's path, a bit ahead of it."""
        if not self._queue:
            self._log("no arm is moving, nothing to drop the object in front of")
            return
        arm, _ = self._queue[0]
        qs = [q for _, q in self._queue]
        for ahead in range(len(qs) // 2, len(qs)):
            spot = arm.fkine(qs[ahead]).t
            landing = [("dropped object", Box("dropped object", spot, DroppedObject.SIZE))]
            if all(first_collision(self._body(a, a.q), [], self._held_spheres(a, a.q), landing) is None
                   for a in (self.arm1, self.arm2)):
                self._scene.hazard.place(spot)
                self._log(f"object dropped in {arm.name}'s path, {ahead} waypoints ahead")
                return
        self._log(f"{arm.name} is too close to the end of its move to drop the object ahead of it")

    @property
    def waypoints_queued(self):
        return len(self._queue)

    def too_close_to_obstacle(self):
        """True if an arm is closer to the dropped object than allowed."""
        hazard = [o for o in self._obstacles() if o[0] == "dropped object"]
        return any(first_collision(self._body(arm, arm.q), [], self._held_spheres(arm, arm.q), hazard) is not None
                   for arm in (self.arm1, self.arm2))

    def aim_obstacle(self, axis, value):
        """Choose where the object gets dropped (moves it if it is already there)."""
        self._hazard_xyz[axis] = value
        if self._scene.hazard.placed:
            self._scene.hazard.place(self._hazard_xyz, fall=False)

    def place_obstacle(self):
        self._scene.hazard.place(self._hazard_xyz)
        self._log(f"object dropped at {np.round(self._hazard_xyz, 2)}")

    def remove_obstacle(self):
        if self._scene.hazard.placed:
            self._scene.hazard.stow()
            self._log("object removed")

    def _path_still_clear(self):
        """Check the planned path again when the object moves.
        Go around it if we can, otherwise stop."""
        hazard = self._scene.hazard
        if self._hazard_checked == hazard.version:
            return True
        self._hazard_checked = hazard.version
        arm = self._queue[0][0]
        remaining = list(self._queue)
        hit = self._first_collision(arm, [q for _, q in remaining])
        if hit is None:
            return True
        if self._transit_goal is not None:
            goal = self._transit_goal
            self._queue.clear()
            try:
                self._move_joint_q(arm, goal)
                self._log(f"upcoming collision ({hit}) -> re-routed around it")
                return True
            except PlanningError:
                self._queue.extend(remaining)
        self._protective_stop(f"upcoming collision ({hit})")
        return False

    # --- manual (teach) mode ------------------------------------------------
    #
    # One arm at a time, picked with select_jog_arm(). Every jog is planned
    # from where the arm is to where it was asked to go, collision-checked
    # like the automatic moves, and played in real time at reduced speed
    # (MANUAL_* above). A jog that would hit something does not move at all.

    def select_jog_arm(self, index):
        arm = (self.arm1, self.arm2)[int(index)]
        if arm is not self.jog_arm:
            self._queue.clear()
            self.jog_arm = arm
            self.jog_version += 1
            self._log(f"jogging {arm.name}")

    def set_jog_enabled(self, enabled):
        self.jog_enabled = bool(enabled)
        if not self.jog_enabled:
            self._queue.clear()
            self.jog_version += 1
        self._log("jog enabled" if self.jog_enabled else "jog disabled")

    def jog_goal(self):
        """Where the jogged arm is heading (where it is, once it gets there)."""
        return self._last_q(self.jog_arm)

    def _can_jog(self):
        if not self.manual:
            self._log("switch to Manual mode to jog")
        elif self.stopped:
            self._log("still stopped - press Acknowledge first")
        elif not self.jog_enabled:
            self._log("tick 'Enable jog' first")
        else:
            return True
        return False

    def _start_jog(self, plan, what):
        """Replace any jog still playing with a new one, planned from where
        the arm is right now. plan(q_now) returns the waypoints."""
        arm = self.jog_arm
        self._queue.clear()
        try:
            qs = plan(arm.q)
        except PlanningError as e:
            self._log(f"{what} refused: {e}")
            self.jog_version += 1           # sliders go back to where the arm is
            return False
        self._enqueue(arm, qs)
        return True

    def jog_joint(self, joint, degrees):
        """Joint slider: move one joint of the jogged arm to `degrees`."""
        if not self._can_jog():
            self.jog_version += 1
            return
        arm = self.jog_arm
        goal = self.jog_goal().copy()
        target = np.clip(np.deg2rad(float(degrees)), *arm.qlim[:, joint])
        if abs(goal[joint] - target) < 1e-4:
            return
        goal[joint] = target

        def plan(q_now):
            qs = self._timed_joint_move(arm, q_now, goal)
            hit = self._first_collision(arm, qs)
            if hit is not None:
                raise PlanningError(hit)
            return qs
        self._start_jog(plan, f"J{joint + 1} jog")

    def jog_home(self):
        """Take the jogged arm back to its home (parked) pose, re-routing round
        anything in the way like the automatic moves do."""
        if not self._can_jog():
            return
        arm = self.jog_arm
        home = ARM1_PARK if arm is self.arm1 else self._home2

        def plan(q_now):
            return self._joint_path(arm, _nearest(arm, home, q_now), q0=q_now)
        if self._start_jog(plan, "go home"):
            self.jog_version += 1
            self._log(f"{arm.name} going home")

    def jog_gripper(self, close):
        """Open or close the jogged arm's gripper. Closing on a pot or the
        can's handle picks it up; opening lets go of what it holds, which
        then settles onto whatever is under it."""
        if not self._can_jog():
            return
        arm = self.jog_arm
        if not close:
            if self._holder is arm:
                obj = self._held
                self._holder = self._held = None
                self._settle(obj)
                self._log(f"{arm.name} let go")
            self._set_finger_goal(arm, FINGER_OFFSET if arm is self.arm1 else REBOT_FINGER_OPEN)
            return
        if self._holder is arm:
            return
        obj, width = self._graspable(arm)
        if obj is None:
            self._set_finger_goal(arm, ARM1_FINGER_CLOSED if arm is self.arm1 else REBOT_FINGER_CLOSED)
            self._log(f"{arm.name} gripper closed (nothing in it)")
            return
        if arm is self.arm2 and width > REBOT_FINGER_OPEN:
            self._log(f"that is wider than the {arm.name} gripper opens")
            return
        self._T_tool_obj = arm.fkine(arm.q).inv() * obj.T
        self._holder, self._held = arm, obj
        self._set_finger_goal(arm, width)
        self._log(f"{arm.name} picked up the {self._object_name(obj)}")

    def _set_finger_goal(self, arm, value):
        if arm is self.arm1:
            self._arm1_finger_goal = value
        else:
            self._finger_goal = value

    def _object_name(self, obj):
        return obj.profile.pot_id if isinstance(obj, Pot) else "watering can"

    def _graspable(self, arm):
        """The pot or can the gripper is in position to grab, and how far
        each finger closes to hold it: a pot when the TCP is inside its wall
        and between its base and rim, the can when the TCP is at its handle."""
        tcp = arm.fkine(arm.q).t
        for pot in self.pots.values():
            if pot is self._held:
                continue
            r, h = pot.profile.diameter_mm / 2000, pot.profile.height_mm / 1000
            base = pot.T.t
            if (np.hypot(*(tcp - base)[:2]) <= r + GRAB_TOLERANCE
                    and base[2] - GRAB_TOLERANCE <= tcp[2] <= base[2] + h + GRAB_TOLERANCE):
                return pot, r
        if self.can is not self._held:
            bar = transform_point(self.can.T, self.can.handle_bar_local)
            if np.linalg.norm(tcp - bar) <= 0.03 + GRAB_TOLERANCE:
                return self.can, WateringCanDuck.HANDLE_WIDTH / 2
        return None, 0.0

    def _object_near_gripper(self, arm, q):
        """In manual mode the gripper may reach into the pot or can nearest
        it (to pick it up), as the automatic moves do with their work object."""
        tcp = arm.fkine(q).t
        near = [(np.linalg.norm((pot.T.t + [0, 0, pot.profile.height_mm / 2000]) - tcp), pot_id)
                for pot_id, pot in self.pots.items() if pot is not self._held]
        if self.can is not self._held:
            near.append((np.linalg.norm(transform_point(self.can.T, self.can.handle_bar_local) - tcp),
                         "watering can"))
        near = [n for n in near if n[0] <= MANUAL_WORK_REACH]
        return min(near)[1] if near else None

    def _settle(self, obj):
        """A let-go object drops straight down onto the highest surface
        under it (shelf, platform, can holder, desk or floor) and stands
        upright there, keeping its heading."""
        x, y, z = obj.T.t
        surfaces = [0.0]
        for centre, size, top in ((TABLE_CENTRE.t, TABLE_SIZE, TABLE_HEIGHT), (SHELF_CENTRE.t, SHELF_SIZE, SHELF_HEIGHT)):
            if abs(x - centre[0]) <= size[0] / 2 and abs(y - centre[1]) <= size[1] / 2:
                surfaces.append(top)
        if np.hypot(x - PLATFORM_POSITION.t[0], y - PLATFORM_POSITION.t[1]) <= 0.10:
            surfaces.append(PLATFORM_TOP)
        if np.hypot(x - CAN_HOLDER_POSITION.t[0], y - CAN_HOLDER_POSITION.t[1]) <= 0.06:
            surfaces.append(CAN_HOLDER_TOP)
        top = max(s for s in surfaces if s <= z + 0.01) if any(s <= z + 0.01 for s in surfaces) else z
        heading = atan2(obj.T.R[1, 0], obj.T.R[0, 0])
        obj.set_pose(SE3(x, y, top) * SE3.Rz(heading))

    def _manual_step(self):
        n = self._ticks_now()
        self._move_fingers(max(n, 1))
        if not self._queue or not self.jog_enabled:
            return
        hazard = self._scene.hazard
        if self._hazard_checked != hazard.version:
            self._hazard_checked = hazard.version
            arm = self._queue[0][0]
            hit = self._first_collision(arm, [q for _, q in self._queue])
            if hit is not None:
                self._queue.clear()
                self.jog_version += 1
                self._log(f"jog stopped: upcoming collision ({hit})")
                return
        for _ in range(min(n, len(self._queue))):
            self._play_waypoint()

    # --- main tick ----------------------------------------------------

    def _ticks_now(self):
        """Waypoints to play this tick: the playback speed (real time in
        manual) times the speed override, carried over between ticks so a
        fractional override still averages out right."""
        self._budget += (1 if self.manual else self.speed) * self.override
        n = int(self._budget)
        self._budget -= n
        return n

    def step(self):
        if self.estop_active or self.stop_latched:
            return
        if self.zone_breached:
            return
        if self.manual:
            self._manual_step()
            return
        if self.paused or self.waiting or self.state in (State.FAULT, State.DONE):
            return
        n = self._ticks_now()
        self._move_fingers(n)
        if self._queue:
            if not self._path_still_clear():
                return
            for _ in range(min(n, len(self._queue))):
                self._play_waypoint()
            return
        if self._hold > 0:
            self._hold = max(0, self._hold - n)
            return
        if not self._entered:
            self._entered = True
            state = self.state
            try:
                self._enter(state)
            except PlanningError as e:
                # nothing moved yet, so Resume can plan this step again
                self._fault(str(e))
                self._resume_state, self._entered, self.stop_latched = state, False, True
            return
        self.state = self._next_state()
        self._entered = False

    def _next_state(self):
        if self.state == State.CHECK_MOISTURE and self.current_pot is None:
            # nothing needs water: finish, or (GUI) keep checking until a plant dries out
            return State.CHECK_MOISTURE if self.keep_watching else State.DONE
        if self.state == State.ARM1_HOME:
            return State.CHECK_MOISTURE
        return _ORDER[_ORDER.index(self.state) + 1]

    def _play_waypoint(self):
        arm, q = self._queue.popleft()
        arm.q = q
        if self._holder is not None:
            self._held.set_pose(self._holder.fkine(self._holder.q) * self._T_tool_obj)

    # --- what each state does when it starts ----------------------------

    def _enter(self, state):
        arm1, arm2, can = self.arm1, self.arm2, self.can

        if state == State.CHECK_MOISTURE:
            # the first plant (small, medium, large) at or below its threshold;
            # a watered plant goes to 100%, so it isn't picked again until its
            # level is turned back down
            self.current_pot = None
            for pot_id, pot in self.pots.items():
                if self.needs_water(pot_id):
                    self.current_pot = pot
                    self._all_ok_reported = False
                    self._log(f"{pot_id}: moisture {self.moisture[pot_id]:.0f}% "
                              f"(waters at {pot.profile.moisture_threshold:.0f}% or below) -> watering")
                    return
            if not self._all_ok_reported:
                self._all_ok_reported = True
                self._log("every plant is moist enough: " + ", ".join(
                    f"{pot_id} {m:.0f}%" for pot_id, m in self.moisture.items()))

        elif state == State.ARM1_TO_SHELF:
            pot = self.current_pot
            grip = pot.T.t + [0, 0, self._grasp_height(pot)]
            self._T_shelf = side_grasp(grip, self._arm1_facing(grip), ARM1_PITCH)
            self._move_joint(arm1, back(self._T_shelf, ARM1_BACKOFF))
        elif state == State.ARM1_APPROACH_POT:
            self._move_line(arm1, self._T_shelf)
        elif state == State.ARM1_GRASP:
            self._grasp(arm1, self.current_pot)
        elif state == State.ARM1_LIFT_FROM_SHELF:
            self._move_line(arm1, up(self._T_shelf, LIFT), SET_DOWN_SPEED)        # ease it off the shelf
            self._move_line(arm1, back(up(self._T_shelf, LIFT), ARM1_BACKOFF))
            self._move_line(arm1, self._shelf_transit())
        elif state == State.ARM1_TO_PLATFORM:
            self._carry_to_platform()
        elif state == State.ARM1_LOWER_ONTO_PLATFORM:
            self._move_line(arm1, self._T_platform, SET_DOWN_SPEED)        # set it down gently
        elif state == State.ARM1_RELEASE:
            self._release()
        elif state == State.ARM1_RETREAT:
            self._move_line(arm1, back(self._T_platform, ARM1_BACKOFF))
            self._move_joint_q(arm1, ARM1_PARK)

        elif state == State.ARM2_GET_CAN:
            # grab the handle bar from the side: one finger over it, one under
            bar_centre = transform_point(can.T, can.handle_bar_local)
            bar_direction = can.T.R @ can.handle_bar_direction_local
            across = np.cross([0.0, 0.0, 1.0], bar_direction)
            if np.dot(across, bar_centre - arm2.base.t) < 0:
                across = -across
            self._T_can = side_grasp(bar_centre + ARM2_GRIP_DEPTH * across, across) * SE3.Rz(np.pi / 2)
            self._move_joint(arm2, back(self._T_can, ARM2_BACKOFF))
        elif state == State.ARM2_APPROACH_CAN:
            self._move_line(arm2, self._T_can)
        elif state == State.ARM2_GRASP_CAN:
            self._grasp(arm2, can)
        elif state == State.ARM2_LIFT_CAN:
            self._move_line(arm2, up(self._T_can, CARRY_LIFT))
        elif state == State.ARM2_TO_POUR_POSITION:
            self._carry_qs, self._lower_qs, self._pour_qs = self._plan_pour()
            self._enqueue(arm2, self._carry_qs + self._lower_qs)
        elif state == State.ARM2_POUR:
            volume = self.current_pot.profile.water_volume_ml
            self._enqueue(arm2, self._pour_qs)
            self._hold += int(volume / POUR_RATE_ML_S / DT)
            self._log(f"pouring {volume:.0f} ml")
        elif state == State.ARM2_UNTILT:
            pot_id = self.current_pot.profile.pot_id
            self.moisture[pot_id] = 100.0       # watered
            self._log(f"{pot_id} watered: moisture now 100%")
            self._enqueue(arm2, reversed(self._lower_qs + self._pour_qs))
        elif state == State.ARM2_RETURN_CAN:
            self._enqueue(arm2, reversed(self._carry_qs))     # same way back, so the can stays upright
            self._move_line(arm2, self._T_can, LOWER_SPEED)
        elif state == State.ARM2_RELEASE_CAN:
            self._release()
        elif state == State.ARM2_HOME:
            self._move_line(arm2, back(self._T_can, ARM2_BACKOFF))
            self._move_joint_q(arm2, self._home2)

        elif state == State.ARM1_TO_PLATFORM_COLLECT:
            self._move_joint(arm1, back(self._T_platform, ARM1_BACKOFF))
            self._move_line(arm1, self._T_platform)
        elif state == State.ARM1_GRASP_PLANT:
            self._grasp(arm1, self.current_pot)
        elif state == State.ARM1_RETURN_TO_SHELF:
            above_slot = up(self._T_shelf, LIFT)
            self._move_line(arm1, self._over_platform(), LOWER_SPEED)          # ease it off the platform
            self._move_arc(arm1, self._shelf_transit(), height_change=CARRY_RISE)
            self._move_line(arm1, back(above_slot, ARM1_BACKOFF))
            self._move_line(arm1, above_slot)
            self._move_line(arm1, self._T_shelf, SET_DOWN_SPEED)           # set it down gently
        elif state == State.ARM1_RELEASE_ON_SHELF:
            self._release()
        elif state == State.ARM1_HOME:
            self._move_line(arm1, back(self._T_shelf, ARM1_BACKOFF))
            self._move_joint_q(arm1, ARM1_PARK)

        elif state == State.DONE:
            self._log("sequence complete")

    # --- moisture ------------------------------------------------------------

    def _fresh_moisture(self):
        """A reading for every plant: the simulated sensor's random value, or
        --moisture for all of them."""
        return {pot_id: float(self.force_moisture if self.force_moisture is not None else read_moisture(pot_id))
                for pot_id in self.pots}

    def needs_water(self, pot_id):
        return self.moisture[pot_id] <= self.pots[pot_id].profile.moisture_threshold

    def set_moisture(self, pot_id, percent):
        """Set a plant's moisture by hand (GUI slider). Turning a watered
        plant back down to its threshold queues it up again."""
        self.moisture[pot_id] = float(np.clip(percent, 0.0, 100.0))
        self._log(f"{pot_id} moisture set to {self.moisture[pot_id]:.0f}%"
                  + (" - needs water" if self.needs_water(pot_id) else ""))

    def _arm1_facing(self, point, offset=0.0):
        """Horizontal direction for arm 1 to grip `point` from: along the arm's
        own plane (offset = 0), so the wrist needs no sideways twist (J4 = J6 = 0)."""
        h = self.arm1.arm_heading(point) + offset
        return np.array([np.cos(h), np.sin(h), 0.0])

    def _carry_to_platform(self):
        """Swing the pot round to the platform and lower it, choosing which
        side arm 1 grips it from there. Coming at the platform head-on
        (gripper pointing straight out from the arm), the TX2-60's wrist must
        pass through its singularity on the way down from the shelf, and J4/J6
        whip round half a turn. The pot is round, so it can just as well be set
        down gripped from an angle. Try approach angles from head-on outwards
        and keep the first with no collision and little wrist roll."""
        arm1 = self.arm1
        grip = np.array([*PLATFORM_POSITION.t[:2], PLATFORM_TOP + self._grasp_height(self.current_pot)])
        problem = None
        for offset in np.deg2rad(PLATFORM_APPROACH_OFFSETS):
            self._T_platform = side_grasp(grip, self._arm1_facing(grip, offset), ARM1_PITCH)
            try:
                swing = self._checked_arc(arm1, self._last_q(arm1), self._over_platform(), CARRY_DROP)
                down = self._checked_line(arm1, swing[-1], up(self._T_platform, LIFT), LOWER_SPEED)
                # the moves that follow from this side have to be clear too:
                # setting the pot down, then backing away from it once released
                set_down = cartesian_move(arm1, down[-1], self._T_platform, SET_DOWN_SPEED)
                pot, carried_pose = self._held, self._held.T
                pot.set_pose(self._T_platform * self._T_tool_obj)       # where it will have been put down
                self._held = self._holder = None
                try:
                    away = self._checked_line(arm1, set_down[-1], back(self._T_platform, ARM1_BACKOFF))
                    self._joint_path(arm1, _nearest(arm1, ARM1_PARK, away[-1]), q0=away[-1])
                finally:
                    self._held, self._holder = pot, arm1
                    pot.set_pose(carried_pose)
            except PlanningError as e:
                problem = str(e)
                continue
            roll = np.rad2deg(np.abs(np.diff(np.array(swing + down)[:, [3, 5]], axis=0)).sum())
            if roll > MAX_CARRY_WRIST_ROLL:
                problem = f"wrist would roll {roll:.0f} deg"
                continue
            self._enqueue(arm1, swing + down)
            return
        raise PlanningError(f"{arm1.name}: no clean way to set the pot on the platform (last problem: {problem})")

    def _shelf_transit(self) -> SE3:
        """Pot lifted clear of the shelf, then pulled straight back
        (horizontally) towards arm 1, ready to swing round."""
        towards_arm = -horizontal_unit(self._T_shelf.R[:, 2])
        return SE3(SHELF_RETREAT * towards_arm) * up(self._T_shelf, SHELF_CLEARANCE)

    def _over_platform(self) -> SE3:
        """Arm 1's platform grasp pose, hovering PLATFORM_HOVER above it."""
        return up(self._T_platform, PLATFORM_HOVER)

    @staticmethod
    def _grasp_height(pot):
        """Grip height above the pot's base. The TX2-60 grips angled down at
        ARM1_PITCH, so its wrist sits above and behind the fingers: gripping
        no lower than ARM1_GRIP_BELOW_RIM under the rim keeps the wrist clear
        of a tall pot's rim, and no lower than ARM1_MIN_GRIP_HEIGHT keeps the
        fingertips clear of the shelf."""
        h = pot.profile.height_mm / 1000
        return max(h + pot.profile.grasp_offset_z, h - ARM1_GRIP_BELOW_RIM, ARM1_MIN_GRIP_HEIGHT)

    def _plan_pour(self):
        """Plan the pour for arm 2. Returns three joint paths: the carry to above
        the pot, the drop to the pour start, and the tilt.

        The spout tip sits just inside the pot's rim and stays there while the
        can tilts around it. For short pots the tip comes down as the can tilts,
        so the can's body does not hit the platform.

        Tries every direction around the pot and keeps the first that works."""
        profile = self.current_pot.profile
        pot_xy = PLATFORM_POSITION.t[:2]
        rim_z = PLATFORM_TOP + profile.pour_clearance_z
        rim_offset = profile.diameter_mm / 2000 - POUR_RIM_INSET

        R_can_in_tool = self._T_tool_obj.R
        tip_tool = transform_point(self._T_tool_obj, self.can.spout_tip_local)

        def tilt_by(angle):
            return SE3.Rt(R_can_in_tool @ SE3.Ry(angle).R @ R_can_in_tool.T, [0, 0, 0], check=False)

        # corners of the can body, measured from the spout tip
        half_l, half_w = WateringCanDuck.BODY_LENGTH / 2, WateringCanDuck.BODY_WIDTH / 2
        corners = np.array([[x, y, z] for x in (-half_l, half_l) for y in (-half_w, half_w)
                            for z in (0.0, WateringCanDuck.BODY_HEIGHT)]) - self.can.spout_tip_local

        def pour_poses(R_carry):
            spout = horizontal_unit(R_carry @ R_can_in_tool @ [1.0, 0.0, 0.0])
            tip_xy = pot_xy - rim_offset * spout[:2]
            poses = []
            for angle in np.linspace(0.0, POUR_ANGLE, POUR_STEPS):
                R = R_carry @ tilt_by(angle).R
                drop = -(corners @ (R @ R_can_in_tool).T)[:, 2].min()   # how far the body hangs below the tip
                tip = np.array([*tip_xy, max(rim_z, PLATFORM_TOP + POUR_FLOOR_MARGIN + drop)])
                poses.append(SE3.Rt(R, tip - R @ tip_tool, check=False))
            return poses

        candidates = []
        for heading in np.deg2rad(np.arange(-180, 180, 10)):
            R_carry = SE3.Rz(heading).R @ self._T_can.R
            if not can_stays_upright(R_carry, R_can_in_tool, tilt_by, 1, POUR_ANGLE):
                continue
            poses = pour_poses(R_carry)
            outward = horizontal_unit(poses[0].t - self.arm2.base.t)
            candidates.append((-np.dot(horizontal_unit(R_carry[:, 2]), outward), poses))
        candidates.sort(key=lambda c: c[0])

        last_reason = "every heading would tip the can past horizontal"
        for _, poses in candidates:
            try:
                q_above = solve_ik(self.arm2, up(poses[0], POUR_APPROACH_LIFT), self._last_q(self.arm2), slimit=50)
                carry_qs = self._carry_path(self.arm2, q_above)
                lower_qs = self._checked_line(self.arm2, q_above, poses[0])
                pour_qs = follow_poses(self.arm2, lower_qs[-1], poses)
                hit = self._first_collision(self.arm2, pour_qs)
                if hit is not None:
                    raise PlanningError(f"{self.arm2.name}: pour tilt collides ({hit})")
            except PlanningError as e:
                last_reason = str(e)
                continue
            return carry_qs, lower_qs, pour_qs
        raise PlanningError(f"{self.arm2.name}: no reachable, collision-free, upright pour pose over "
                            f"{profile.pot_id} (last problem: {last_reason})")

    # --- queueing motion --------------------------------------------------

    def _last_q(self, arm):
        for a, q in reversed(self._queue):
            if a is arm:
                return q
        return arm.q

    def _seed(self, arm, T_goal: SE3):
        if arm is self.arm1:
            a = T_goal.R[:, 2]
            return np.array([atan2(a[1], a[0]), 0.0, np.pi / 2, 0.0, 0.0, 0.0])
        return self._last_q(arm)

    def _move_joint(self, arm, T_goal: SE3):
        # seeded from a tidy pose so the IK converges, then turned into the
        # equivalent nearest where the arm actually is (_move_joint_q)
        self._move_joint_q(arm, solve_ik(arm, T_goal, self._seed(arm, T_goal)))

    def _enqueue(self, arm, qs, transit_goal=None):
        self._queue.extend((arm, q) for q in qs)
        self._transit_goal = transit_goal

    def _move_joint_q(self, arm, q_goal):
        q_goal = _nearest(arm, q_goal, self._last_q(arm))
        self._enqueue(arm, self._joint_path(arm, q_goal), transit_goal=q_goal)

    def _move_line(self, arm, T_goal: SE3, speed=TCP_SPEED):
        qs = self._checked_line(arm, self._last_q(arm), T_goal, speed)
        self._enqueue(arm, qs)

    def _move_arc(self, arm, T_goal: SE3, height_change=(0.0, 1.0)):
        self._enqueue(arm, self._checked_arc(arm, self._last_q(arm), T_goal, height_change))

    def _checked_arc(self, arm, q_from, T_goal: SE3, height_change=(0.0, 1.0)):
        qs = arc_move(arm, q_from, T_goal, height_change=height_change)
        hit = self._first_collision(arm, qs)
        if hit is not None:
            raise PlanningError(f"{arm.name}: carry arc collides ({hit})")
        return qs

    def _carry_path(self, arm, q_goal):
        """Path to q_goal that keeps the full can upright. Tries a straight
        joint move first, then level swings around the base, each one higher."""
        q0 = self._last_q(arm)
        can_in_tool = self._T_tool_obj.R
        reason = ""
        plans = [("joint move", lambda: self._timed_joint_move(arm, q0, q_goal))]
        for hump in CARRY_HUMPS:
            plans.append((f"level swing +{hump * 100:.0f} cm",
                          lambda hump=hump: upright_move(arm, q0, arm.fkine(q_goal), hump)))
        for name, plan in plans:
            try:
                qs = plan()
            except PlanningError as e:
                reason = str(e)
                continue
            lean = max(np.arccos(np.clip((arm.fkine(q).R @ can_in_tool)[2, 2], -1, 1)) for q in qs)
            if lean > CARRY_MAX_TILT:
                reason = f"{name} would tip the can {np.rad2deg(lean):.0f} deg"
                continue
            hit = self._first_collision(arm, qs)
            if hit is None:
                return qs
            reason = f"{name} collides ({hit})"
        raise PlanningError(f"{arm.name}: cannot carry the can upright ({reason})")

    def _joint_path(self, arm, q_goal, q0=None):
        q0 = self._last_q(arm) if q0 is None else q0
        home = _nearest(arm, ARM1_PARK if arm is self.arm1 else self._home2, q0)
        tucked_here = np.r_[q0[0], home[1:]]
        tucked_there = np.r_[q_goal[0], home[1:]]
        hit = None
        for route in ([q0, q_goal], [q0, home, q_goal], [q0, tucked_here, tucked_there, q_goal]):
            path = [q for a, b in zip(route, route[1:]) for q in self._timed_joint_move(arm, a, b)]
            hit = self._first_collision(arm, path)
            if hit is None:
                return path
        raise PlanningError(f"{arm.name}: every route collides ({hit})")

    def _timed_joint_move(self, arm, q_from, q_to):
        """Joint move under both the joint speed limit and JOINT_MOVE_TCP_SPEED,
        so a big swing doesn't fling the gripper round much faster than the
        straight-line moves either side of it."""
        qs = joint_move(q_from, q_to, self._joint_speed(arm))
        if len(qs) > 1:
            p = arm.fkine(np.array(qs)).t
            fastest = np.linalg.norm(np.diff(p, axis=0), axis=1).max() / DT
            cap = MANUAL_TCP_SPEED if self.manual else JOINT_MOVE_TCP_SPEED
            if fastest > cap:
                steps = int(np.ceil(len(qs) * fastest / cap))
                qs = list(rtb.jtraj(q_from, q_to, steps).q)
        return qs

    def _joint_speed(self, arm):
        """Top speed per joint (rad/s) for free joint-space moves."""
        share = MANUAL_JOINT_SPEED_FRACTION / JOINT_SPEED_FRACTION if self.manual else 1.0
        if hasattr(arm, "qd_max"):
            return share * JOINT_SPEED_FRACTION * arm.qd_max
        return np.full(arm.n, share * REBOT_JOINT_SPEED)

    def _checked_line(self, arm, q_from, T_goal: SE3, speed=TCP_SPEED):
        qs = cartesian_move(arm, q_from, T_goal, speed)
        hit = self._first_collision(arm, qs)
        if hit is not None:
            raise PlanningError(f"{arm.name}: straight-line move collides ({hit})")
        return qs

    # --- collision checking -----------------------------------------------

    def _first_collision(self, arm, qs):
        other = self.arm2 if arm is self.arm1 else self.arm1
        others = self._body(other, self._last_q(other))
        obstacles = self._obstacles()
        if self.manual:
            work = self._object_near_gripper(arm, qs[-1]) if len(qs) else None
        else:
            work = self.current_pot.profile.pot_id if arm is self.arm1 and self.current_pot else "watering can"
        for q in qs:
            hit = first_collision(self._body(arm, q), others, self._held_spheres(arm, q), obstacles, work)
            if hit is not None:
                return hit
        return None

    def _body(self, arm, q):
        if arm is self.arm1:
            return tx2_60_body(self._dh1, q, GRIPPER_LENGTH, FINGER_OFFSET)
        return rebot_body(arm, q)

    def _obstacles(self):
        obs = list(self._furniture)
        hazard = self._scene.hazard
        if hazard.placed:
            obs.append(("dropped object", Box("dropped object", hazard.position, DroppedObject.SIZE)))
        for pot_id, pot in self.pots.items():
            if pot is not self._held:
                obs.append((pot_id, VCylinder(pot_id, pot.T.t, pot.profile.diameter_mm / 2000,
                                              pot.profile.height_mm / 1000)))
        if self.can is not self._held:
            # the whole can, handle included, as one box
            base, can = self.can.T.t, WateringCanDuck
            obs.append(("watering can", Box("watering can", base + [0, 0, can.BODY_HEIGHT / 2], can.BODY_SIZE)))
        return obs

    def _held_spheres(self, arm, q):
        if self._holder is not arm:
            return None
        T = arm.fkine(q) * self._T_tool_obj
        if isinstance(self._held, Pot):
            p = self._held.profile
            return cylinder(p.pot_id, T, p.diameter_mm / 2000, 0.0, p.height_mm / 1000)
        # can body as two columns of spheres
        can = WateringCanDuck
        r = can.BODY_WIDTH / 2
        halves = [cylinder("watering can", T * SE3(side * (can.BODY_LENGTH / 2 - r), 0, 0), r,
                           0.0, can.BODY_HEIGHT) for side in (1, -1)]
        root = transform_point(T, self.can.bill_root_local)
        spout = capsule("spout", root, transform_point(T, self.can.spout_tip_local), 0.01)
        parts = halves + [spout]
        return Spheres("watering can", np.vstack([p.centres for p in parts]),
                       np.concatenate([p.radii for p in parts]))

    # --- gripper ------------------------------------------------------------

    def _grasp(self, arm, obj):
        self._T_tool_obj = arm.fkine(arm.q).inv() * obj.T
        self._holder, self._held = arm, obj
        if arm is self.arm2:
            self._finger_goal = WateringCanDuck.HANDLE_WIDTH / 2
        if arm is self.arm1:
            self._arm1_finger_goal = obj.profile.diameter_mm / 2000   # close onto the pot wall
        self._hold += GRIP_TICKS

    def _release(self):
        if self._holder is self.arm2:
            self._finger_goal = REBOT_FINGER_OPEN
        if self._holder is self.arm1:
            self._arm1_finger_goal = FINGER_OFFSET
        self._holder = self._held = None
        self._hold += GRIP_TICKS

    def _move_fingers(self, n):
        """Both grippers close onto / open off what they hold over the grip pause."""
        step = n * REBOT_FINGER_OPEN / GRIP_TICKS
        gap = self._finger_goal - self._scene.arm2_finger_opening
        self._scene.arm2_finger_opening += float(np.clip(gap, -step, step))
        # TX2-60: same pace, from fully open onto the pot over the grip pause
        step1 = n * FINGER_OFFSET / GRIP_TICKS
        gap1 = self._arm1_finger_goal - self.arm1.finger_offset
        if gap1:
            self.arm1.set_finger_offset(self.arm1.finger_offset + float(np.clip(gap1, -step1, step1)))

    def _log(self, msg):
        print(f"[{'MANUAL' if self.manual else self.state.name}] {msg}")


# ---------------------------------------------------------------------
# Scripted collision test
# ---------------------------------------------------------------------

class CollisionTest:
    """Scripted test (--collision-test). Drops an object in front of arm 1 twice:
    once on a free move (it should go around) and once while it carries a plant
    (it should stop, then carry on after Acknowledge + Resume)."""

    CASES = [(State.ARM1_TO_SHELF, "re-route"), (State.ARM1_TO_PLATFORM, "stop")]
    MIN_WAYPOINTS = 25      # only drop while this much of the move is left
    PAUSE_TICKS = 40        # ticks to stay stopped before removing the object

    def __init__(self, seq):
        self.seq = seq
        self.cases = list(self.CASES)
        self.active = None
        self.wait = 0
        self.results = []
        self.got_too_close = False

    @property
    def idle(self):
        return self.active is None

    def tick(self):
        seq = self.seq
        self.got_too_close = self.got_too_close or seq.too_close_to_obstacle()
        if self.active is None:
            if self.cases and seq.state == self.cases[0][0] and seq.waypoints_queued >= self.MIN_WAYPOINTS:
                self.active = self.cases.pop(0)
                self.wait = self.PAUSE_TICKS
                seq.drop_obstacle_ahead()
            return
        state, _ = self.active
        if seq.state == State.FAULT:
            self.wait -= 1
            if self.wait <= 0:
                self._finish("stop")
                seq.remove_obstacle()
                seq.acknowledge()
                seq.resume()
        elif seq.state != state:
            self._finish("re-route")
            seq.remove_obstacle()

    def _finish(self, response):
        state, expected = self.active
        self.results.append((state.name, expected, response))
        self.active = None

    def report(self):
        print("Collision test:")
        for state, expected, response in self.results:
            print(f"  object dropped during {state}: expected {expected}, got {response}")
        for state, expected in self.cases:
            print(f"  object never dropped during {state.name} (the sequence did not get there)")
        print(f"  an arm came inside its clearance from the object: {'YES' if self.got_too_close else 'no'}")
        passed = (not self.cases and not self.got_too_close and self.seq.state == State.DONE
                  and all(expected == response for _, expected, response in self.results))
        print("  RESULT:", "PASS" if passed else "FAIL")


# ---------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ShelfCare watering sequence")
    parser.add_argument("--headless", action="store_true",
                        help="plan and run every move without opening Swift")
    parser.add_argument("--moisture", type=float, default=None,
                        help="force every moisture reading to this %%, e.g. 10 to water every plant")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED,
                        help=f"playback speed-up, 1 = real time (default {DEFAULT_SPEED})")
    parser.add_argument("--collision-test", action="store_true",
                        help="drop an object into an arm's path twice and report how the sequence responds")
    parser.add_argument("--paused", action="store_true",
                        help="no longer needed: Swift always opens in the starting state and waits for "
                             "Start / Resume")
    args = parser.parse_args()

    env = None
    if not args.headless:
        import swift
        env = swift.Swift()
        env.launch(realtime=True)

    scene = build_scene(env, POT_REGISTRY.values())
    seq = ShelfCareSequence(scene, force_moisture=args.moisture, speed=args.speed)
    test = CollisionTest(seq) if args.collision_test else None

    if env is None:
        while seq.state != State.DONE:
            seq.step()
            if test is not None:
                test.tick()
            if seq.state == State.FAULT and (test is None or test.idle):
                break
        report(seq, test)
        return

    # With the GUI the loop keeps running after the sequence ends, so the
    # buttons (Reset in particular) still work. Close it with Ctrl+C.
    from control_panel import ControlPanel

    seq.waiting = True          # nothing moves while the page loads: wait for Start / Resume
    seq.keep_watching = test is None    # never 'done' (water any plant turned down again), except in the test
    panel, reported = ControlPanel(env, seq), False
    while True:
        seq.step()
        if test is not None:
            test.tick()
        panel.update()
        if seq.state == State.DONE and not reported:
            report(seq, test)
        reported = seq.state == State.DONE
        scene.safety.show(seq.signal, seq.zone_breached)
        scene.refresh()
        env.step(DT)


def report(seq, test):
    print("Final state:", seq.state.name, seq.fault_reason if seq.state == State.FAULT else "")
    if test is not None:
        test.report()


if __name__ == "__main__":
    main()
