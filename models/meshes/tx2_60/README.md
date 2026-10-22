Stäubli TX2-60 visual meshes, unmodified, from ROS-Industrial staubli_experimental
(commit 256a71a, Apache License 2.0, see LICENSE):
https://github.com/ros-industrial/staubli_experimental

    base_link.stl, link_1.stl ... link_4.stl   staubli_tx2_60_support/meshes/tx2_60/visual/
    link_5.stl, link_6.stl                     staubli_tx60_support/meshes/tx60/visual/
                                               (the TX2-60 URDF uses the TX60 wrist meshes)

Units are metres. Each mesh is in its link's frame from tx2_60_macro.xacro, which is
the same chain StaubliTX260 in models/staubli_tx2_60.py is built from.
