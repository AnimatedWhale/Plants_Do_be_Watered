"""Visual twin of the reBot DH model: the real URDF meshes, posed from the DH joint angles. Drawing only."""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3


def urdf_mesh_colours(urdf_path):
    """Colour of each visual mesh file, from the URDF's materials."""
    root = ET.parse(urdf_path).getroot()
    rgba = {m.get("name"): [float(v) for v in m.find("color").get("rgba").split()]
            for m in root.findall("material") if m.find("color") is not None}
    colours = {}
    for visual in root.iter("visual"):
        mesh, material = visual.find("geometry/mesh"), visual.find("material")
        if mesh is not None and material is not None and material.get("name") in rgba:
            colours[os.path.basename(mesh.get("filename"))] = rgba[material.get("name")]
    return colours

def load_urdf(urdf_path):
    """Load the URDF without its collision meshes (the sim checks collisions with its own shapes)."""
    try:
        from roboticstoolbox.models.URDF.URDFRobot import URDF_file
    except ImportError:
        return rtb.ERobot.URDF(urdf_path)
    drop_collision = lambda text: re.sub(r"<collision>.*?</collision>", "", text, flags=re.S)
    links, name, _ = URDF_file(urdf_path, patch=drop_collision)
    return rtb.Robot(links, name=name)


class ReBotVisual:
    """The URDF reBot for Swift. Call update() each tick with the DH joint angles."""
    def __init__(self, urdf_path: str, dh_tool: SE3, base: SE3 = None):
        self.robot = load_urdf(urdf_path)
        # the toolbox loads every part grey, so paint each one with its URDF material
        colours = urdf_mesh_colours(urdf_path)
        for link in self.robot.links:
            for shape in link.geometry:
                name = os.path.basename(shape.filename or "")
                if name in colours:
                    shape.color = colours[name]
        if base is not None:
            self.robot.base = base
        self._dh_tool = dh_tool
        self.robot.q = np.zeros(self.robot.n)

    def update(self, dh_robot, dh_q, finger_opening: float = 0.0):
        """Copy the DH joint angles onto the mesh (the URDF joints match the DH ones).
        finger_opening: 0 = closed, up to 0.05 m."""
        self.robot.q = np.r_[np.asarray(dh_q)[:6], finger_opening, -finger_opening]

    def flange_error(self, dh_robot, dh_q) -> float:
        """Distance (m) between the mesh's flange and the DH model's. Should be about 0."""
        flange = dh_robot.fkine(dh_q) * self._dh_tool.inv()
        return float(np.linalg.norm(self.robot.fkine(self.robot.q, end="end_link").t - flange.t))

    def add_to(self, env):
        env.add(self.robot)
