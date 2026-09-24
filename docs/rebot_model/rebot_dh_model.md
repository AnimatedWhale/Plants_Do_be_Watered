# reBot B601-DM — DH parameter model
```python
import numpy as np
from math import pi
import roboticstoolbox as rtb

phi = np.arctan2(0.054, 0.2426)
a3 = np.hypot(0.2426, 0.054)

class ReBotB601DM(rtb.DHRobot):
    def __init__(self):
        links = [
            rtb.RevoluteDH(d=0.1402,    a=0.020084, alpha=pi/2,  offset=0,       qlim=[-2.8, 2.8]),   # J1 base/waist
            rtb.RevoluteDH(d=-0.031625, a=0.264,    alpha=pi,    offset=pi,      qlim=[-3.14, 0]),    # J2 shoulder
            rtb.RevoluteDH(d=-0.001625, a=a3,       alpha=0,     offset=pi-phi,  qlim=[-3.14, 0]),    # J3 elbow
            rtb.RevoluteDH(d=-0.030,    a=0.078308, alpha=-pi/2, offset=phi,     qlim=[-1.87, 1.57]), # J4 forearm bend
            rtb.RevoluteDH(d=0.0025,    a=0,        alpha=pi/2,  offset=pi/2,    qlim=[-1.57, 1.57]), # J5 wrist pitch
            rtb.RevoluteDH(d=0.183398,  a=0,        alpha=0,     offset=0,       qlim=[-3.14, 3.14]), # J6 wrist roll
        ]
        super().__init__(links, name="reBot B601-DM")
```