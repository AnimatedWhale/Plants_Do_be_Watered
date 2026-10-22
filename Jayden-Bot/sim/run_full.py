"""
Run ShelfCare with the full-detail reBot meshes (the heavier render).

Same program as Selfcare_sequence.py; this only picks which reBot meshes to draw.
Any option works here too, e.g.
    python Jayden-Bot/sim/run_full.py --moisture 10
"""

import swift_scene_setup

swift_scene_setup.REBOT_URDF_PATH = swift_scene_setup.REBOT_URDF_FULL

from Selfcare_sequence import main  # noqa: E402  (imported after the mesh choice is made)

if __name__ == "__main__":
    main()
