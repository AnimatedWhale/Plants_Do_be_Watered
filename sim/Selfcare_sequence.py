"""
ShelfCare sequence skeleton.

Two arms:
  ARM 1 (Stäubli TX2-60)   - window sill pick / place on platform / return
  ARM 2 (reBot B601-DM)    - picks up the watering can, pours into the pot

Each pot has known geometry (diameter, height, sill slot) so the grasp
point, the platform drop height, and the pour position/volume all scale
correctly instead of being hardcoded for one pot.
"""

from dataclasses import dataclass
from enum import Enum, auto
import random


# ---------------------------------------------------------------------
# Pot geometry registry
# ---------------------------------------------------------------------

@dataclass
class PotProfile:
    pot_id: str
    diameter_mm: float          # rim diameter, sets gripper opening + pour offset
    height_mm: float            # pot height, sets grasp height + pour clearance
    sill_position: tuple        # (x, y, z) on the window sill, world frame, metres
    moisture_threshold: float   # % below which the plant needs watering
    water_volume_ml: float      # how much to pour, scaled to pot size

    @property
    def grasp_offset_z(self) -> float:
        """How far down from the pot rim the gripper should close, in metres.
        Keeps grip below the rim regardless of pot height."""
        return -0.6 * (self.height_mm / 1000)

    @property
    def pour_clearance_z(self) -> float:
        """Height above the rim the watering can spout should sit when
        pouring, so taller pots get a higher pour point automatically."""
        return (self.height_mm / 1000) + 0.05  # 5 cm clearance above rim


# Fixed platform position all pots get placed at (arm 1 places here,
# arm 2 pours here, arm 1 collects here)
PLATFORM_POSITION = (0.30, 0.00, 0.10)  # TODO: match your Swift scene

# Registry of known pots. Extend this as you add more to the sill.
POT_REGISTRY = {
    "pot_small": PotProfile(
        pot_id="pot_small", diameter_mm=80, height_mm=90,
        sill_position=(0.10, 0.40, 0.35),
        moisture_threshold=30.0, water_volume_ml=100,
    ),
    "pot_medium": PotProfile(
        pot_id="pot_medium", diameter_mm=120, height_mm=140,
        sill_position=(0.10, 0.40, 0.55),
        moisture_threshold=35.0, water_volume_ml=200,
    ),
    "pot_large": PotProfile(
        pot_id="pot_large", diameter_mm=160, height_mm=190,
        sill_position=(0.10, 0.40, 0.75),
        moisture_threshold=40.0, water_volume_ml=350,
    ),
}



# Simulated moisture sensor


def read_moisture(pot_id: str) -> float:
    """Returns a simulated moisture reading (0-100%) for a pot.
    TODO: replace with a real sensor read if you wire one up for the
    PLC bonus, or keep this simulated call and trigger it from a GUI
    button/slider so a marker can force a 'needs watering' state."""
    return random.uniform(10, 60)


def needs_watering(pot: PotProfile) -> bool:
    return read_moisture(pot.pot_id) < pot.moisture_threshold



# Sequence state machine


class State(Enum):
    IDLE = auto()
    CHECK_MOISTURE = auto()
    ARM1_TO_SILL = auto()
    ARM1_GRASP = auto()
    ARM1_TO_PLATFORM = auto()
    ARM1_RELEASE = auto()
    ARM2_GET_CAN = auto()
    ARM2_TO_POUR_POSITION = auto()
    ARM2_POUR = auto()
    ARM2_RETURN_CAN = auto()
    ARM1_TO_PLATFORM_COLLECT = auto()
    ARM1_GRASP_PLANT = auto()
    ARM1_RETURN_TO_SILL = auto()
    DONE = auto()
    FAULT = auto()


class ShelfCareSequence:
    def __init__(self):
        self.state = State.IDLE
        self.current_pot: PotProfile | None = None
        self.estop_active = False

    def estop(self):
        """Called by the GUI e-stop button or the physical e-stop input.
        Latches into FAULT; does not auto-resume when released."""
        self.estop_active = True
        self.state = State.FAULT

    def reset_and_resume(self):
        """Deliberate two-step recovery: reset clears the fault flag,
        a SEPARATE resume call is needed to actually continue."""
        if self.state == State.FAULT:
            self.estop_active = False
            # caller must call resume() next; do not auto-continue here

    def resume(self):
        if not self.estop_active:
            self.state = State.IDLE

    def step(self):
        """Advance one step of the sequence. Call this from your GUI's
        update loop (a timer/thread), never inside a blocking while loop."""
        if self.estop_active:
            return  # latched, do nothing until reset_and_resume + resume

        if self.state == State.IDLE:
            self.state = State.CHECK_MOISTURE

        elif self.state == State.CHECK_MOISTURE:
            for pot in POT_REGISTRY.values():
                if needs_watering(pot):
                    self.current_pot = pot
                    self.state = State.ARM1_TO_SILL
                    return
            self.state = State.DONE  # nothing needs watering this pass

        elif self.state == State.ARM1_TO_SILL:
            # TODO: arm1.move_to(self.current_pot.sill_position)
            self.state = State.ARM1_GRASP

        elif self.state == State.ARM1_GRASP:
            # TODO: arm1.close_gripper(width=self.current_pot.diameter_mm)
            # TODO: grasp at self.current_pot.grasp_offset_z below the rim
            self.state = State.ARM1_TO_PLATFORM

        elif self.state == State.ARM1_TO_PLATFORM:
            # TODO: arm1.move_to(PLATFORM_POSITION) via RMRC/trajectory
            self.state = State.ARM1_RELEASE

        elif self.state == State.ARM1_RELEASE:
            # TODO: arm1.open_gripper(); arm1.retreat()
            self.state = State.ARM2_GET_CAN

        elif self.state == State.ARM2_GET_CAN:
            # TODO: arm2.move_to(can_holder_position); arm2.close_gripper()
            self.state = State.ARM2_TO_POUR_POSITION

        elif self.state == State.ARM2_TO_POUR_POSITION:
            # Pour offset scales with this pot's height/diameter
            x, y, z = PLATFORM_POSITION
            pour_target = (x, y, z + self.current_pot.pour_clearance_z)
            # TODO: arm2.move_to(pour_target)
            self.state = State.ARM2_POUR

        elif self.state == State.ARM2_POUR:
            # TODO: arm2.pour(volume_ml=self.current_pot.water_volume_ml)
            self.state = State.ARM2_RETURN_CAN

        elif self.state == State.ARM2_RETURN_CAN:
            # TODO: arm2.move_to(can_holder_position); arm2.open_gripper()
            self.state = State.ARM1_TO_PLATFORM_COLLECT

        elif self.state == State.ARM1_TO_PLATFORM_COLLECT:
            # TODO: arm1.move_to(PLATFORM_POSITION)
            self.state = State.ARM1_GRASP_PLANT

        elif self.state == State.ARM1_GRASP_PLANT:
            # TODO: arm1.close_gripper(width=self.current_pot.diameter_mm)
            self.state = State.ARM1_RETURN_TO_SILL

        elif self.state == State.ARM1_RETURN_TO_SILL:
            # TODO: arm1.move_to(self.current_pot.sill_position)
            # TODO: arm1.open_gripper()
            self.current_pot = None
            self.state = State.CHECK_MOISTURE  # check the next pot

        elif self.state == State.DONE:
            pass  # sequence complete, GUI can show idle/complete state


if __name__ == "__main__":
    seq = ShelfCareSequence()
    for _ in range(30):
        seq.step()
        if seq.state in (State.DONE, State.FAULT):
            break
    print("Final state:", seq.state)