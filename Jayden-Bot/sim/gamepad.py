"""
Xbox controller for Manual mode.

The controller only moves the arm and has an E-STOP.

  left stick     move the gripper left/right (x) and forward/back (y),
                 or turn the joint picked in the panel (left/right)
  RT / LT        move the gripper up / down (z)
  View button    E-STOP


"""

import os
import time

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")    # keep reading while the browser has focus
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError:
    pygame = None

STICK_X, STICK_Y = 0, 1
LEFT_TRIGGER, RIGHT_TRIGGER = 4, 5
VIEW_BUTTON = 6
DEAD_ZONE = 0.15           


def _dead_zone(v):
    return 0.0 if abs(v) < DEAD_ZONE else (v - DEAD_ZONE * (1 if v > 0 else -1)) / (1 - DEAD_ZONE)


class GamepadJog:
    def __init__(self, seq):
        self.seq = seq
        self.pad = None
        self._next_look = 0.0
        self._view_was_down = False
        self._trigger_seen = [False, False]   
        self.seq.pad_status = "not found" if pygame is None else "looking..."
        if pygame is not None:
            pygame.init()
            pygame.joystick.init()

    def update(self):
        if pygame is None:
            return
        pygame.event.pump()
        if self.pad is None or pygame.joystick.get_count() == 0:
            self._find_pad()
            return
        try:
            x = _dead_zone(self.pad.get_axis(STICK_X))
            y = -_dead_zone(self.pad.get_axis(STICK_Y))         
            z = self._trigger(RIGHT_TRIGGER, 1) - self._trigger(LEFT_TRIGGER, 0)
            view = self.pad.get_button(VIEW_BUTTON)
        except pygame.error:
            self._lost()
            return
        if view and not self._view_was_down:
            self.seq.estop()
        self._view_was_down = view
        self.seq.pad_jog(x, y, z)

    def _trigger(self, axis, index):
        """0 released .. 1 fully pressed."""
        value = self.pad.get_axis(axis)
        if value < -0.5:
            self._trigger_seen[index] = True
        if not self._trigger_seen[index]:
            return 0.0
        return max(0.0, _dead_zone((value + 1) / 2))

    def _find_pad(self):
        if time.monotonic() < self._next_look:
            return
        self._next_look = time.monotonic() + 1.0
        pygame.joystick.quit()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            if self.pad is not None:
                self._lost()
            self.seq.pad_status = "not found"
            return
        self.pad = pygame.joystick.Joystick(0)
        self.pad.init()
        self._trigger_seen = [False, False]
        self.seq.pad_status = f"{self.pad.get_name()} connected"
        print(f"[controller] {self.pad.get_name()} connected")

    def _lost(self):
        self.pad = None
        self.seq.pad_jog(0, 0, 0)
        self.seq.pad_status = "disconnected"
        print("[controller] disconnected")


def show_live_values():
    """Print what the controller reports, to check its layout."""
    if pygame is None:
        print("pygame is not installed: python -m pip install pygame")
        return
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller found. Plug it in and run this again.")
        return
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"{pad.get_name()}: {pad.get_numaxes()} axes, {pad.get_numbuttons()} buttons. Ctrl+C to stop.")
    print(f"Expected: left stick = axes {STICK_X}/{STICK_Y}, LT = axis {LEFT_TRIGGER}, "
          f"RT = axis {RIGHT_TRIGGER}, View = button {VIEW_BUTTON}")
    try:
        while True:
            pygame.event.pump()
            axes = " ".join(f"{i}:{pad.get_axis(i):+.2f}" for i in range(pad.get_numaxes()))
            pressed = [i for i in range(pad.get_numbuttons()) if pad.get_button(i)]
            print(f"\raxes {axes}   buttons pressed {pressed}        ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    show_live_values()
