"""
ShelfCare sequence.

Two arms:
  ARM 1 (Stäubli TX2-60)  - shelf pick / place on platform / return to shelf
  ARM 2 (reBot B601-DM)   - picks up the watering can, pours into the pot

For each plant whose (simulated) moisture is below its threshold:
  1. arm 1 lifts the pot off the shelf and puts it on the platform
  2. arm 2 picks the watering can off its holder, pours over the pot,
     and puts the can back
  3. arm 1 returns the pot to its shelf slot
then the next plant is checked, until none need water.

Each pot has known geometry (diameter, height, shelf slot), so the grasp
point and the pour height scale with the pot instead of being hardcoded.

Free travel uses joint-space moves (jtraj). Approach, lift, retreat and
carrying a pot use straight-line Cartesian moves (ctraj solved point by
point with IK), so the gripper never swings sideways into the shelf, the pot
or the can, and a carried pot stays upright.

Every waypoint of every move is collision-checked (collision.py) against the
desk, wall, shelf, platform, can holder, pots, can and the other arm before
the arm moves. A joint move that would hit something is rerouted through the
arm's home pose; if nothing works the sequence faults and says what would hit.

step() is non-blocking: each call plays one tick. Call it from the GUI /
sim loop, never inside a blocking while loop of its own.

Run from the repo root:
    python Jayden-Bot/sim/Selfcare_sequence.py                # Swift in the browser
    python Jayden-Bot/sim/Selfcare_sequence.py --moisture 10  # force every plant to need water
    python Jayden-Bot/sim/Selfcare_sequence.py --speed 1      # real-time playback (default 4x)
    python Jayden-Bot/sim/Selfcare_sequence.py --headless     # plan and run every move without a browser
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

from swift_scene_setup import ARM1_PARK, PLATFORM_POSITION, PLATFORM_TOP, Pot, WateringCan, build_scene
from collision import VCylinder, Spheres, capsule, cylinder, first_collision, rebot_body, tx2_60_body
from models.staubli_tx2_60 import FINGER_OFFSET, GRIPPER_LENGTH, StaubliTX260DH


# ---------------------------------------------------------------------
# Pot geometry registry
# ---------------------------------------------------------------------

@dataclass
class PotProfile:
    pot_id: str
    diameter_mm: float          # rim diameter, sets gripper opening
    height_mm: float            # pot height, sets grasp height + pour clearance
    shelf_position: tuple       # (x, y) of the pot's slot on the shelf, world frame, metres
    moisture_threshold: float   # % below which the plant needs watering
    water_volume_ml: float      # how much to pour, scaled to pot size

    @property
    def grasp_offset_z(self) -> float:
        """How far down from the pot rim the gripper should close, in metres.
        Keeps grip below the rim regardless of pot height."""
        return -0.6 * (self.height_mm / 1000)

    @property
    def pour_clearance_z(self) -> float:
        """Height above the pot's base the spout tip sits at when pouring:
        10 cm above the rim, so taller pots get a higher pour point and the
        spout clears the foliage."""
        return (self.height_mm / 1000) + 0.10


# Registry of known pots. Extend this as you add more to the shelf; the
# shelf slots must match the shelf in swift_scene_setup.py.
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

DT = 0.05                       # s per simulation tick (motion is planned at this rate)
DEFAULT_SPEED = 5               # playback speed-up: waypoints played per tick
JOINT_STEPS = 60                # ticks for a free joint-space move (3 s)
CART_STEPS = 30                 # ticks for a short straight-line move (1.5 s)
CARRY_STEPS = 80                # ticks for carrying a pot shelf <-> platform in a straight line (4 s)
GRIP_TICKS = 10                 # gripper open/close time (0.5 s)
ARM1_BACKOFF = 0.15             # m, pre-grasp distance back along the approach axis (clears the widest pot)
ARM2_BACKOFF = 0.08             # m, same for arm 2 and the can
ARM1_PITCH = np.deg2rad(30)     # arm 1 grips pots angled down, keeping J5 clear of its limit at the low platform
ARM2_PITCH = np.deg2rad(30)     # the reBot reaches far more side grasps angled down than level
LIFT = 0.04                     # m, lift clear of the shelf / platform / holder before moving
SHELF_CLEARANCE = 0.30          # m, carried pot's base rises this far above the shelf before
                                #   swinging round, so it passes over the other plants
CARRY_LIFT = 0.10               # m, arm 2 lift after picking the can up
POUR_ANGLE = np.deg2rad(80)     # wrist roll while pouring; with the 30 deg grip this tips the spout well below level
POUR_RATE_ML_S = 50.0           # sets how long the can stays tilted
MAX_JOINT_STEP = 0.3            # rad per tick, for arms without rated joint speeds; more means a config flip

DOWN = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------
# Simulated moisture sensor
# ---------------------------------------------------------------------

def read_moisture(pot_id: str) -> float:
    """Returns a simulated moisture reading (0-100%) for a pot.
    TODO: replace with a real sensor read if you wire one up for the
    PLC bonus, or keep this simulated call and trigger it from a GUI
    button/slider so a marker can force a 'needs watering' state."""
    return random.uniform(10, 60)


# ---------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------

class PlanningError(RuntimeError):
    """IK failed, the arm would flip configuration mid-move, or a move would collide."""


def horizontal_unit(v):
    v = np.array([v[0], v[1], 0.0])
    return v / np.linalg.norm(v)


def side_grasp(point, approach, pitch=0.0) -> SE3:
    """TCP pose for a side grasp: tool z = approach direction, tilted down
    by `pitch` from horizontal; tool x = 'down' square to that; fingers close
    along tool y, which stays horizontal."""
    h = horizontal_unit(approach)
    a = np.cos(pitch) * h + np.sin(pitch) * DOWN
    x = np.cos(pitch) * DOWN - np.sin(pitch) * h
    return SE3(point) * SE3.OA(np.cross(a, x), a)


def transform_point(T: SE3, p):
    return (T.A @ np.append(p, 1.0))[:3]


def _ik_goal(arm, T_goal: SE3) -> SE3:
    """The goal in the frame this robot's ikine expects. Depending on the
    roboticstoolbox version and robot class, ikine either includes arm.base
    or leaves it out (1.4: ERobot leaves it out, DHRobot includes it), so
    find out once per robot by solving for a pose it is already at."""
    if not hasattr(arm, "_ik_in_base_frame"):
        T_here = arm.fkine(arm.q)
        sol = arm.ikine_LM(T_here, q0=arm.q, tol=1e-10, slimit=1)
        arm._ik_in_base_frame = not (sol.success and np.abs(arm.fkine(sol.q).A - T_here.A).max() < 1e-6)
    return arm.base.inv() * T_goal if arm._ik_in_base_frame else T_goal


def solve_ik(arm, T_goal: SE3, q0, slimit=100):
    # The default tol stops ~1 mm short, hence 1e-10; fkine (which always
    # includes base and tool) confirms the answer really lands on the goal.
    sol = arm.ikine_LM(_ik_goal(arm, T_goal), q0=q0, joint_limits=True, tol=1e-10, slimit=slimit)
    if sol.success and np.abs(arm.fkine(sol.q).A - T_goal.A).max() < 1e-3:
        return sol.q
    raise PlanningError(f"{arm.name}: no IK solution for TCP at {np.round(T_goal.t, 3)}")


def joint_move(q_from, q_to, steps=JOINT_STEPS):
    return list(rtb.jtraj(q_from, q_to, steps).q)


def cartesian_move(arm, q_from, T_to: SE3, steps=CART_STEPS):
    """Straight-line TCP path."""
    return follow_poses(arm, q_from, rtb.ctraj(arm.fkine(q_from), T_to, steps))


def arc_move(arm, q_from, T_to: SE3, steps=CARRY_STEPS):
    """TCP path that swings round the arm's base axis, blending radius,
    height and heading, while the tool turns only about the vertical. Used
    to carry a pot: it stays upright and at a comfortable reach instead of
    cutting the corner past the arm's own base like a straight line would."""
    T_from = arm.fkine(q_from)
    c = arm.base.t
    p0, p1 = T_from.t - c, T_to.t - c
    r0, r1 = np.hypot(*p0[:2]), np.hypot(*p1[:2])
    a0 = atan2(p0[1], p0[0])
    da = (atan2(p1[1], p1[0]) - a0 + np.pi) % (2 * np.pi) - np.pi
    poses = []
    for t in np.linspace(0.0, 1.0, steps):
        s = (1 - np.cos(np.pi * t)) / 2          # smooth start and stop
        a, r, z = a0 + s * da, r0 + s * (r1 - r0), p0[2] + s * (p1[2] - p0[2])
        R = SE3.Rz(s * da).R @ T_from.R
        poses.append(SE3.Rt(R, c + [r * np.cos(a), r * np.sin(a), z]))
    return follow_poses(arm, q_from, poses)


def follow_poses(arm, q_from, poses):
    """IK for each pose in turn, each seeded from the previous waypoint."""
    # Biggest allowed change per tick: the joint's rated speed if the model
    # has one (TX2-60), otherwise MAX_JOINT_STEP. Anything more is a flip.
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
    """The grasp pose moved back along its approach (tool z) axis."""
    return T_grasp * SE3(0, 0, -distance)


def up(T: SE3, height) -> SE3:
    """The pose moved straight up in the world frame."""
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


# Normal running order; FAULT is outside it.
_ORDER = [s for s in State if s is not State.FAULT]


class ShelfCareSequence:
    def __init__(self, scene, force_moisture=None, speed=DEFAULT_SPEED):
        self.arm1, self.arm2, self.can, self.pots = scene.arm1, scene.arm2, scene.can, scene.pots
        self.force_moisture = force_moisture
        self.speed = max(1, int(speed))
        self._furniture = scene.obstacles
        self._dh1 = StaubliTX260DH()          # DH twin of arm 1 for the collision model
        self._dh1.base = self.arm1.base

        self.state = State.IDLE
        self.estop_active = False
        self.fault_reason = ""
        self._resume_state = None

        self.current_pot = None   # scene Pot being watered
        self._checked = set()     # pot_ids already handled this run

        self._queue = deque()     # (arm, q) waypoints still to play
        self._hold = 0            # ticks to wait (gripper, pouring)
        self._entered = False     # has the current state planned its motion yet?
        self._holder = None       # arm currently holding something
        self._held = None         # the Pot or WateringCan it holds
        self._T_tool_obj = None   # held object's pose in the holder's TCP frame
        self._home2 = self.arm2.q.copy()

    # --- safety -------------------------------------------------------

    def estop(self):
        """GUI e-stop button or physical e-stop input. Latches into FAULT
        and does not auto-resume when released."""
        if self.estop_active or self.state == State.DONE:
            return
        self.estop_active = True
        if self.state != State.FAULT:
            self._resume_state = self.state
        self.state = State.FAULT
        self._log("E-STOP")

    def reset(self):
        """Step 1 of recovery: clear the latched e-stop. The arms stay
        stopped until resume() is called separately."""
        if self.state == State.FAULT:
            self.estop_active = False

    reset_and_resume = reset  # name used in the original skeleton

    def resume(self):
        """Step 2 of recovery: carry on from exactly where the e-stop hit."""
        if self.state == State.FAULT and not self.estop_active and self._resume_state is not None:
            self.state, self._resume_state = self._resume_state, None
            self._log("resumed")

    def _fault(self, reason):
        self._log(f"FAULT: {reason}")
        self._queue.clear()
        self.fault_reason = f"{self.state.name}: {reason}"
        self._resume_state = None   # a planning fault can't be resumed
        self.state = State.FAULT

    # --- main tick ----------------------------------------------------

    def step(self):
        """Advance one tick (`speed` planned waypoints). Call this from your
        GUI's update loop."""
        if self.estop_active or self.state in (State.FAULT, State.DONE):
            return
        if self._queue:
            for _ in range(min(self.speed, len(self._queue))):
                self._play_waypoint()
            return
        if self._hold > 0:
            self._hold = max(0, self._hold - self.speed)
            return
        if not self._entered:
            self._entered = True
            try:
                self._enter(self.state)
            except PlanningError as e:
                self._fault(str(e))
            return
        self.state = self._next_state()
        self._entered = False

    def _next_state(self):
        if self.state == State.CHECK_MOISTURE and self.current_pot is None:
            return State.DONE
        if self.state == State.ARM1_HOME:
            return State.CHECK_MOISTURE   # check the next pot
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
            self.current_pot = None
            for pot_id, pot in self.pots.items():
                if pot_id in self._checked:
                    continue
                self._checked.add(pot_id)
                m = self.force_moisture if self.force_moisture is not None else read_moisture(pot_id)
                needs = m < pot.profile.moisture_threshold
                self._log(f"{pot_id}: moisture {m:.1f}% (threshold {pot.profile.moisture_threshold}%)"
                          + (" -> watering" if needs else ""))
                if needs:
                    self.current_pot = pot
                    return

        # Arm 1: shelf -> platform
        elif state == State.ARM1_TO_SHELF:
            pot = self.current_pot
            grip = pot.T.t + [0, 0, self._grasp_height(pot)]
            self._T_shelf = side_grasp(grip, grip - arm1.base.t, ARM1_PITCH)
            self._move_joint(arm1, back(self._T_shelf, ARM1_BACKOFF))
        elif state == State.ARM1_APPROACH_POT:
            self._move_line(arm1, self._T_shelf)
        elif state == State.ARM1_GRASP:
            # TODO: real gripper close, width = self.current_pot.profile.diameter_mm
            self._grasp(arm1, self.current_pot)
        elif state == State.ARM1_LIFT_FROM_SHELF:
            self._move_line(arm1, up(self._T_shelf, LIFT))
            self._move_line(arm1, back(up(self._T_shelf, LIFT), ARM1_BACKOFF))
            self._move_line(arm1, back(up(self._T_shelf, SHELF_CLEARANCE), ARM1_BACKOFF))
        elif state == State.ARM1_TO_PLATFORM:
            # Swing round at the lifted height (an arc, not a joint move, so
            # the pot stays upright), then straight down over the platform
            grip = np.array([*PLATFORM_POSITION.t[:2], PLATFORM_TOP + self._grasp_height(self.current_pot)])
            self._T_platform = side_grasp(grip, grip - arm1.base.t, ARM1_PITCH)
            self._move_arc(arm1, self._over_platform())
            self._move_line(arm1, up(self._T_platform, LIFT), CARRY_STEPS)
        elif state == State.ARM1_LOWER_ONTO_PLATFORM:
            self._move_line(arm1, self._T_platform)
        elif state == State.ARM1_RELEASE:
            self._release()
        elif state == State.ARM1_RETREAT:
            self._move_line(arm1, back(self._T_platform, ARM1_BACKOFF))
            self._move_joint_q(arm1, ARM1_PARK)

        # Arm 2: holder -> pour -> holder
        elif state == State.ARM2_GET_CAN:
            # Grip the can from arm 2's side, square to the spout, so tilting
            # about the approach axis (a pure wrist roll) tips the spout down.
            spout = can.T.R @ [1.0, 0.0, 0.0]
            side = np.array([-spout[1], spout[0], 0.0])
            if np.dot(side, can.T.t - arm2.base.t) < 0:
                side = -side
            self._T_can = side_grasp(can.T.t, side, ARM2_PITCH)
            self._move_joint(arm2, back(self._T_can, ARM2_BACKOFF))
        elif state == State.ARM2_APPROACH_CAN:
            self._move_line(arm2, self._T_can)
        elif state == State.ARM2_GRASP_CAN:
            self._grasp(arm2, can)
        elif state == State.ARM2_LIFT_CAN:
            self._move_line(arm2, up(self._T_can, CARRY_LIFT))
        elif state == State.ARM2_TO_POUR_POSITION:
            self._T_carry, self._T_pour = self._plan_pour()
            self._move_joint(arm2, self._T_carry)
        elif state == State.ARM2_POUR:
            volume = self.current_pot.profile.water_volume_ml
            self._move_line(arm2, self._T_pour)
            self._hold += int(volume / POUR_RATE_ML_S / DT)
            self._log(f"pouring {volume:.0f} ml")
        elif state == State.ARM2_UNTILT:
            self._move_line(arm2, self._T_carry)
        elif state == State.ARM2_RETURN_CAN:
            self._move_joint(arm2, up(self._T_can, CARRY_LIFT))
            self._move_line(arm2, self._T_can)
        elif state == State.ARM2_RELEASE_CAN:
            self._release()
        elif state == State.ARM2_HOME:
            self._move_line(arm2, back(self._T_can, ARM2_BACKOFF))
            self._move_joint_q(arm2, self._home2)

        # Arm 1: platform -> shelf. The pot hasn't moved since arm 1 let go,
        # so the platform grasp pose is still valid.
        elif state == State.ARM1_TO_PLATFORM_COLLECT:
            self._move_joint(arm1, back(self._T_platform, ARM1_BACKOFF))
            self._move_line(arm1, self._T_platform)
        elif state == State.ARM1_GRASP_PLANT:
            self._grasp(arm1, self.current_pot)
        elif state == State.ARM1_RETURN_TO_SHELF:
            above_slot = up(self._T_shelf, LIFT)
            self._move_line(arm1, self._over_platform(), CARRY_STEPS)
            self._move_arc(arm1, back(up(self._T_shelf, SHELF_CLEARANCE), ARM1_BACKOFF))
            self._move_line(arm1, back(above_slot, ARM1_BACKOFF))
            self._move_line(arm1, above_slot)
            self._move_line(arm1, self._T_shelf)
        elif state == State.ARM1_RELEASE_ON_SHELF:
            self._release()
        elif state == State.ARM1_HOME:
            self._move_line(arm1, back(self._T_shelf, ARM1_BACKOFF))
            self._move_joint_q(arm1, ARM1_PARK)
            # current_pot stays set until CHECK_MOISTURE picks the next one, so
            # the gripper backing away still counts as working on this pot

        elif state == State.DONE:
            self._log("sequence complete")

    def _over_platform(self) -> SE3:
        """Arm 1's platform grasp pose raised to the shelf transit height."""
        transit_z = back(up(self._T_shelf, SHELF_CLEARANCE), ARM1_BACKOFF).t[2]
        return up(self._T_platform, transit_z - self._T_platform.t[2])

    @staticmethod
    def _grasp_height(pot):
        """Grip height above the pot's base, from the profile's offset below the rim."""
        return pot.profile.height_mm / 1000 + pot.profile.grasp_offset_z

    def _plan_pour(self):
        """Carry pose (can upright) and pour pose (can tilted) for arm 2.

        The spout can come at the pot from any side (any heading about the
        vertical). Arm 2 reaches best with its gripper pointing away from its
        own base, so try headings in 10 deg steps, most 'pointing away' first,
        and keep the first where arm 2 reaches both poses without hitting
        anything and can tilt between them in a straight line."""
        target_xy = PLATFORM_POSITION.t[:2]

        # Tool x points (roughly) down, so rolling about tool z moves the spout
        # (which lies along +/- tool y) downwards for one sign of the angle.
        spout_tool = self._T_tool_obj.R @ [1.0, 0.0, 0.0]
        tilt = SE3.Rz(-np.sign(spout_tool[1]) * POUR_ANGLE)

        # Put the TCP where, once tilted, the spout tip is over the pot centre;
        # the height scales with this pot's pour clearance.
        tip_tool = transform_point(self._T_tool_obj, self.can.spout_tip_local)
        tip_target = np.array([*target_xy, PLATFORM_TOP + self.current_pot.profile.pour_clearance_z])

        candidates = []
        for heading in np.deg2rad(np.arange(-180, 180, 10)):
            R_carry = SE3.Rz(heading).R @ self._T_can.R
            T_carry = SE3.Rt(R_carry, tip_target - R_carry @ tilt.R @ tip_tool)
            outward = horizontal_unit(T_carry.t - self.arm2.base.t)
            candidates.append((-np.dot(horizontal_unit(R_carry[:, 2]), outward), T_carry))
        candidates.sort(key=lambda c: c[0])

        last_reason = ""
        for _, T_carry in candidates:
            try:
                q_carry = solve_ik(self.arm2, T_carry, self._last_q(self.arm2), slimit=50)
                self._joint_path(self.arm2, q_carry)
                self._checked_line(self.arm2, q_carry, T_carry * tilt)
            except PlanningError as e:
                last_reason = str(e)
                continue
            return T_carry, T_carry * tilt
        raise PlanningError(f"{self.arm2.name}: no reachable, collision-free pour pose over "
                            f"{self.current_pot.profile.pot_id} (last problem: {last_reason})")

    # --- queueing motion --------------------------------------------------

    def _last_q(self, arm):
        """Where the arm will be once everything already queued has played."""
        for a, q in reversed(self._queue):
            if a is arm:
                return q
        return arm.q

    def _seed(self, arm, T_goal: SE3):
        if arm is self.arm1:
            # Elbow-up TX2-60 pose facing the goal: the IK then lands on a tidy
            # configuration instead of whatever is nearest the current one.
            a = T_goal.R[:, 2]
            return np.array([atan2(a[1], a[0]), 0.0, np.pi / 2, 0.0, 0.0, 0.0])
        return self._last_q(arm)

    def _move_joint(self, arm, T_goal: SE3):
        self._move_joint_q(arm, solve_ik(arm, T_goal, self._seed(arm, T_goal)))

    def _move_joint_q(self, arm, q_goal):
        self._queue.extend((arm, q) for q in self._joint_path(arm, q_goal))

    def _move_line(self, arm, T_goal: SE3, steps=CART_STEPS):
        qs = self._checked_line(arm, self._last_q(arm), T_goal, steps)
        self._queue.extend((arm, q) for q in qs)

    def _move_arc(self, arm, T_goal: SE3):
        qs = arc_move(arm, self._last_q(arm), T_goal)
        hit = self._first_collision(arm, qs)
        if hit is not None:
            raise PlanningError(f"{arm.name}: carry arc collides ({hit})")
        self._queue.extend((arm, q) for q in qs)

    def _joint_path(self, arm, q_goal):
        """Collision-free joint-space path to q_goal. Routes tried in order:
          1. straight there
          2. via the arm's home pose
          3. tuck into the home posture, swing the base (J1) round to face
             the goal while tucked, then reach out"""
        q0 = self._last_q(arm)
        home = ARM1_PARK if arm is self.arm1 else self._home2
        tucked_here = np.r_[q0[0], home[1:]]
        tucked_there = np.r_[q_goal[0], home[1:]]
        hit = None
        for route in ([q0, q_goal], [q0, home, q_goal], [q0, tucked_here, tucked_there, q_goal]):
            path = [q for a, b in zip(route, route[1:]) for q in joint_move(a, b)]
            hit = self._first_collision(arm, path)
            if hit is None:
                return path
        raise PlanningError(f"{arm.name}: every route collides ({hit})")

    def _checked_line(self, arm, q_from, T_goal: SE3, steps=CART_STEPS):
        qs = cartesian_move(arm, q_from, T_goal, steps)
        hit = self._first_collision(arm, qs)
        if hit is not None:
            raise PlanningError(f"{arm.name}: straight-line move collides ({hit})")
        return qs

    # --- collision checking -----------------------------------------------

    def _first_collision(self, arm, qs):
        """First collision along a list of joint configurations for `arm`,
        with the other arm wherever its queued motion leaves it."""
        other = self.arm2 if arm is self.arm1 else self.arm1
        others = self._body(other, self._last_q(other))
        obstacles = self._obstacles()
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
        """Furniture plus every pot and the can, except whatever is being held."""
        obs = list(self._furniture)
        for pot_id, pot in self.pots.items():
            if pot is not self._held:
                obs.append((pot_id, VCylinder(pot_id, pot.T.t, pot.profile.diameter_mm / 2000,
                                              pot.profile.height_mm / 1000)))
        if self.can is not self._held:
            base = self.can.T.t - [0, 0, WateringCan.HEIGHT / 2]
            obs.append(("watering can", VCylinder("watering can", base, WateringCan.RADIUS, WateringCan.HEIGHT)))
        return obs

    def _held_spheres(self, arm, q):
        """The held object as spheres, at the pose it would have with the arm at q."""
        if self._holder is not arm:
            return None
        T = arm.fkine(q) * self._T_tool_obj
        if isinstance(self._held, Pot):
            p = self._held.profile
            return cylinder(p.pot_id, T, p.diameter_mm / 2000, 0.0, p.height_mm / 1000)
        body = cylinder("watering can", T, WateringCan.RADIUS, -WateringCan.HEIGHT / 2, WateringCan.HEIGHT / 2)
        root = transform_point(T, [WateringCan.RADIUS, 0.0, 0.02])
        spout = capsule("spout", root, transform_point(T, self.can.spout_tip_local), 0.01)
        return Spheres("watering can", np.vstack([body.centres, spout.centres]),
                       np.concatenate([body.radii, spout.radii]))

    # --- gripper ------------------------------------------------------------

    def _grasp(self, arm, obj):
        # TODO: call the real gripper close here
        self._T_tool_obj = arm.fkine(arm.q).inv() * obj.T
        self._holder, self._held = arm, obj
        self._hold += GRIP_TICKS

    def _release(self):
        # TODO: call the real gripper open here
        self._holder = self._held = None
        self._hold += GRIP_TICKS

    def _log(self, msg):
        print(f"[{self.state.name}] {msg}")


# ---------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------

def add_controls(env, seq):
    """E-stop / reset / resume buttons and a state readout in the Swift page."""
    import swift

    status = swift.Label(desc="State: IDLE")
    env.add(status)
    env.add(swift.Button(desc="E-STOP", cb=lambda *_: seq.estop()))
    env.add(swift.Button(desc="Reset", cb=lambda *_: seq.reset()))
    env.add(swift.Button(desc="Resume", cb=lambda *_: seq.resume()))
    return status


def main():
    parser = argparse.ArgumentParser(description="ShelfCare watering sequence")
    parser.add_argument("--headless", action="store_true",
                        help="plan and run every move without opening Swift")
    parser.add_argument("--moisture", type=float, default=None,
                        help="force every moisture reading to this %%, e.g. 10 to water every plant")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED,
                        help=f"playback speed-up, 1 = real time (default {DEFAULT_SPEED})")
    args = parser.parse_args()

    env = None
    if not args.headless:
        import swift
        env = swift.Swift()
        env.launch(realtime=True)

    scene = build_scene(env, POT_REGISTRY.values())
    seq = ShelfCareSequence(scene, force_moisture=args.moisture, speed=args.speed)

    status, shown = (add_controls(env, seq), None) if env is not None else (None, None)
    while seq.state != State.DONE:
        seq.step()
        if env is None:
            if seq.state == State.FAULT:
                break   # headless: nobody to press reset
            continue
        if seq.state != shown:
            shown = seq.state
            status.desc = f"State: {shown.name}"
        scene.refresh()
        env.step(DT)

    print("Final state:", seq.state.name, seq.fault_reason)
    if env is not None:
        env.hold()


if __name__ == "__main__":
    main()
