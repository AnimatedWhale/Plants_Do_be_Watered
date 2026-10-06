# Stäubli TX2-60: DH parameters

Hand-drawn full DH diagram: `dh_full_diagram.jpg` (to be added).

Reference frame diagram:

![TX2-60 DH frames](tx2_60_dh_reference.svg)

## Link dimensions

| Symbol | Value | Meaning |
|---|---|---|
| d1 | 375 mm | base floor to the shoulder (J2) axis |
| a2 | 290 mm | shoulder (J2) to elbow (J3), upper arm |
| d3 | 20 mm | sideways offset between J2 and J3, along the J2 axis |
| d4 | 310 mm | elbow (J3) to wrist centre (J5), forearm |
| d6 | 70 mm | wrist centre to tool flange |

Sanity check: a2 + d4 + d6 = 290 + 310 + 70 = **670 mm**, which is the reach Stäubli publishes for the TX2-60 [2][3].

## Standard DH table (used by the code)

T<sub>i</sub> = Rz(θ<sub>i</sub>) · Tz(d<sub>i</sub>) · Tx(a<sub>i</sub>) · Rx(α<sub>i</sub>)

| i | θ<sub>i</sub> | d<sub>i</sub> (m) | a<sub>i</sub> (m) | α<sub>i</sub> | Joint limits [1] |
|---|---|---|---|---|---|
| 1 | q1 | 0.375 | 0 | −π/2 | ±180° |
| 2 | q2 − π/2 | 0 | 0.290 | 0 | ±127.5° |
| 3 | q3 + π/2 | 0.020 | 0 | +π/2 | ±152.5° |
| 4 | q4 | 0.310 | 0 | −π/2 | ±270° |
| 5 | q5 | 0 | 0 | +π/2 | −121° / +132.5° |
| 6 | q6 | 0.070 | 0 | 0 | ±270° |

Tool: `SE3(0, 0, 0.15)` for a 150 mm gripper. The fingers have to reach past the centre of the widest pot (160 mm) and open to 170 mm. Measure yours and change `GRIPPER_LENGTH` / `FINGER_OFFSET` in `models/staubli_tx2_60.py`.

The −π/2 and +π/2 offsets on joints 2 and 3 make **q = 0 the arm standing straight up**. That's the Stäubli zero position, so the angles in the sim match the teach pendant.



## Sources

1. ROS-Industrial, *staubli_experimental*, `staubli_tx2_60_support/urdf/tx2_60_macro.xacro`: joint origins, joint limits and joint speeds. https://github.com/ros-industrial/staubli_experimental
2. Stäubli, *TX2-60 6-axis industrial robot* product page (reach 670 mm). https://www.staubli.com/global/en/robotics/products/industrial-robots/6-axis/tx2-60.html
3. RoboDK, *Staubli TX2-60* robot library entry (reach 670 mm, repeatability 0.02 mm). https://robodk.com/robot/Staubli/TX2-60
4. J. Denavit and R. S. Hartenberg, "A kinematic notation for lower-pair mechanisms based on matrices," *ASME Journal of Applied Mechanics*, 22, pp. 215–221, 1955.
5. P. Corke, *Robotics, Vision and Control: Fundamental Algorithms in Python*, 3rd ed., Springer, 2023. Covers the DH/MDH conventions and the Robotics Toolbox for Python.

For the final report, check the joint limits against the Stäubli TX2-60 instruction manual or datasheet from your lab. The values here come from the ROS-Industrial URDF [1].
