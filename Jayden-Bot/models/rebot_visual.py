"""Mesh is the visual twin of the rebot_b601_dm, mainly used for the swift visuals but does not have an input in the kinematics """

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

class ReBotVisual:
    """wrap the URDF reBot for swift rendering uses call update(q) in each tick to update with the current joint values and then updates them here """
    def __init__(self, urdf_path: str, dh_tool: SE3, base: SE3 = None):
        self.robot = rtb.ERobot.URDF(urdf_path)
        if base is not None:
            self.robot.base = base
        self._dh_tool = dh_tool
        self.robot.q = np.zeros(self.robot.n)

    def update(self, dh_robot, dh_q, finger_opening: float = 0.0):
        """dh_q: its current 6 joint values.
        finger_opening: 0 (closed) to ~0.05 m (open), mirrored on both
        fingers per the URDF's mimic joint.

        The URDF's 6 arm joints share the DH model's zeros and directions
        (end_link sits on the DH flange, i.e. fkine(q) * tool.inv(), to
        within ~0.2 mm), so the DH joint values drive the mesh directly."""
        self.robot.q = np.r_[np.asarray(dh_q)[:6], finger_opening, -finger_opening]

    def flange_error(self, dh_robot, dh_q) -> float:
        """Position gap (m) between the mesh's end_link and the DH flange,
        a sanity check that the mesh still tracks the DH model."""
        flange = dh_robot.fkine(dh_q) * self._dh_tool.inv()
        return float(np.linalg.norm(self.robot.fkine(self.robot.q, end="end_link").t - flange.t))

    def add_to(self, env):
        env.add(self.robot)
