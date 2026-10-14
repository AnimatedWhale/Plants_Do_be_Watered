"""
ShelfCare operator panel for Swift.

Swift stacks its controls top to bottom in the order they are added, so the
panel is laid out in sections under heading labels:

  status      mode + state + what to press
  safety      E-STOP, Acknowledge, Resume, Pause/Hold, Reset, speed override
  mode        Auto / Manual
  manual      arm, enable switch, J1-J6 sliders, X/Y/Z jog, gripper, go home
  collision   the red box: where to drop it, drop it, take it away

Controls that don't apply right now (jogging in Auto, Resume while stopped)
do nothing; the reason is printed in the terminal. All the logic lives in
ShelfCareSequence; this file only wires the controls to it.
"""

import numpy as np
import swift

ECHO_TICKS = 10         # ignore a joint slider for this long after the panel moves it (stale values echo back)


def _number(value):
    """Swift hands some control values over as strings."""
    return float(value[0] if isinstance(value, (list, tuple)) else value)


class ControlPanel:
    def __init__(self, env, seq):
        self.env, self.seq = env, seq
        self._tick = 0
        self._shown = {}
        self._jog_version = None

        self.status = self._label("")

        self._label("------ SAFETY ------")
        self._button("E-STOP", seq.estop)
        self._button("Acknowledge stop", seq.acknowledge)
        self._button("Resume", seq.resume)
        self._button("Pause / Hold", seq.pause)
        self._button("Reset to start", seq.reset)
        self._add(swift.Slider(cb=lambda v: seq.set_override(_number(v)), min=5, max=100, step=5, value=100,
                               desc="Speed override", unit="%"))

        self._label("------ MODE ------")
        self.mode = self._add(swift.Radio(cb=self._on_mode, desc="Mode",
                                          options=["Auto", "Manual"], checked=0))

        self._label("------ MANUAL CONTROL (Manual mode only) ------")
        self._add(swift.Select(cb=lambda i: seq.select_jog_arm(int(_number(i))), desc="Arm",
                               options=[seq.arm1.name, seq.arm2.name], value=0))
        self.enable = self._add(swift.Checkbox(cb=self._on_enable, desc="Enable switch",
                                               options=["Enable jog"], checked=[False]))
        self.joints = [self._add(swift.Slider(cb=lambda v, j=j: self._on_joint(j, v),
                                              min=-180, max=180, step=1, value=0, desc=f"J{j + 1}", unit="°"))
                       for j in range(6)]
        self._joint_shown = [0.0] * 6     # what each joint slider shows in the browser
        self._joint_synced = [0] * 6      # tick the panel last moved each joint slider
        self.jog_step = 0.01
        self._add(swift.Slider(cb=self._on_step, min=1, max=50, step=1, value=10, desc="Jog step", unit="mm"))
        for axis, name in enumerate("XYZ"):
            for sign in (1, -1):
                self._button(f"{name}{'+' if sign > 0 else '-'}",
                             lambda axis=axis, sign=sign: seq.jog_cartesian(axis, sign * self.jog_step))
        self._button("Gripper open", lambda: seq.jog_gripper(False))
        self._button("Gripper close", lambda: seq.jog_gripper(True))
        self._button("Go home", seq.jog_home)

        self._label("------ COLLISION TEST (red box) ------")
        # the sliders aim the object; it enters the cell when a Drop button is pressed
        for axis, (low, high, start) in enumerate(((0.0, 1.4, 0.70), (0.0, 0.7, 0.35), (0.75, 1.5, 1.00))):
            self._add(swift.Slider(cb=lambda v, axis=axis: seq.aim_obstacle(axis, _number(v)),
                                   min=low, max=high, step=0.01, value=start, desc=f"Object {'xyz'[axis]}", unit="m"))
        self._button("Drop object at sliders", seq.place_obstacle)
        self._button("Drop object in path", seq.drop_obstacle_ahead)
        self._button("Remove object", seq.remove_obstacle)

        self.update()

    # --- building -----------------------------------------------------------

    def _add(self, element):
        self.env.add(element)
        return element

    def _label(self, text):
        return self._add(swift.Label(desc=text))

    def _button(self, text, action):
        return self._add(swift.Button(desc=text, cb=lambda *_: action()))

    # Swift's page reports every change to a control back to Python, including
    # changes Python made itself (e.g. ticking Enable to match the sim). So each
    # callback ignores a value the sim is already at, or it would loop forever.

    def _on_mode(self, index):
        manual = int(_number(index)) == 1
        if manual == self.seq.manual:
            return
        self.seq.set_manual(manual)
        if self.seq.manual != manual:       # refused: put the radio back
            self._shown.pop((id(self.mode), "checked"), None)

    def _on_enable(self, checked):
        on = checked[0] if isinstance(checked, (list, tuple)) else checked
        on = on in (True, "true", 1, "1")
        if on != self.seq.jog_enabled:
            self.seq.set_jog_enabled(on)

    def _on_joint(self, j, value):
        value = _number(value)
        if abs(value - self._joint_shown[j]) < 1e-6 or self._tick - self._joint_synced[j] < ECHO_TICKS:
            return
        self._joint_shown[j] = value
        self.seq.jog_joint(j, value)

    def _on_step(self, value):
        self.jog_step = _number(value) / 1000

    # --- keeping it up to date ----------------------------------------------

    def _set(self, element, attr, value):
        """Only send a change to the browser when something actually changed."""
        key = (id(element), attr)
        if self._shown.get(key) != value:
            self._shown[key] = value
            setattr(element, attr, value)

    def update(self):
        """Call once per pass of the main loop."""
        seq = self.seq
        self._set(self.status, "desc", seq.status)

        self._tick += 1

        self._set(self.mode, "checked", [not seq.manual, seq.manual])
        if self._shown.get("enable") != seq.jog_enabled:
            # set it directly: swift's Checkbox prints every value given to its setter
            self._shown["enable"] = seq.jog_enabled
            self.enable._checked, self.enable._changed = [seq.jog_enabled], True

        # point the joint sliders at the jogged arm's goal whenever it changes
        # other than by dragging a slider (new arm, jog refused, X/Y/Z, home...)
        if seq.jog_version != self._jog_version:
            self._jog_version = seq.jog_version
            arm = seq.jog_arm
            low, high = np.rad2deg(arm.qlim)
            goal = np.rad2deg(seq.jog_goal())
            for j, slider in enumerate(self.joints):
                value = float(np.round(goal[j]))
                if (self._joint_shown[j], slider.min, slider.max) != (value, np.floor(low[j]), np.ceil(high[j])):
                    slider.min, slider.max = float(np.floor(low[j])), float(np.ceil(high[j]))
                    self._joint_shown[j] = slider.value = value
                    self._joint_synced[j] = self._tick
