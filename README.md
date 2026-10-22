# Plants_Do_be_Watered
Over engineered Watering Can

Take plant from shelf then place on desk and use another robot arm to water it 

Project description
ShelfCare automates watering plants stored on a wall shelf beyond easy reach. Arm 1 lifts a potted plant from the shelf and places it on a platform. Arm 2, the reBot B601-DM, then picks up a watering can and pours a measured amount into the pot. Once watering is complete, Arm 1 returns the plant to its shelf position. A simulated moisture sensor decides which plants need attention. Safety features include a keep-out zone around the platform during watering, collision detection along both arms' paths, and a physical e-stop on the reBot. A Python GUI lets an operator jog either arm and monitor system state.
Real robot
 reBot B601-DM — used for the watering-can task, since its six-axis reach and gripper payload suit lifting and tilting a filled can accurately.
Each student's robot model

The Stäubli TX2-60, a six-DoF industrial arm absent from both the 41013 catalogue and Corke's toolbox. Its published kinematic dimensions and 5 kg payload comfortably support lifting a potted plant off a shelf and placing it on the watering platform without tipping. 

## Contributions

Rudra (GitHub: Rusa05) and Jayden (GitHub: Jayden Kim / AnimatedWhale). Based on `git blame` line counts.

### Rudra
- `models/staubli_tx2_60.py`: Stäubli TX2-60 model (DH, MDH and Swift models, wrist-flip handling, self-check)
- `models/meshes/tx2_60/`: TX2-60 meshes for the Swift model
- `docs/staubli_model/staubli_dh_model.md`: TX2-60 DH table and write-up
- `Jayden-Bot/sim/control_panel.py`: the operator control panel (GUI)
- `ShelfCare WaterBot-Proposal.md`, `requirements.txt`

### Jayden
- `Jayden-Bot/models/rebot_b601_dm.py`: reBot B601-DM DH model
- `Jayden-Bot/models/rebot_description/`, `Jayden-Bot/models/rebot_visual.py`: reBot URDFs, meshes and their Swift visual
- `Jayden-Bot/sim/collision.py`: collision checking for both arms
- `Jayden-Bot/sim/gamepad.py`: Xbox controller jogging and E-STOP
- `Jayden-Bot/sim/run_lite.py`, `Jayden-Bot/sim/run_full.py`: launchers
- `models/link_visuals.py`, `docs/rebot_model/`

### Shared
- `Jayden-Bot/sim/Selfcare_sequence.py` (Jayden about two thirds, Rudra about one third)
  - Jayden: main state machine, reBot can pick-up and pour, E-STOP / fault handling, light curtain response, re-routing around a dropped object, collision test routine
  - Rudra: move timing (`joint_move`, `cartesian_move`, `arc_move`, `_timed_joint_move`), TX2-60 pot carrying (`_carry_to_platform`, `_shelf_transit`), manual teach mode (joint jogging, gripper, go home), pause / resume / speed override / mode switch, moisture sliders
- `Jayden-Bot/sim/swift_scene_setup.py` (about half each)
  - Jayden: room, safety equipment, watering can, scene layout
  - Rudra: desk, shelf, platform, can holder, pots, obstacle list, most of `build_scene`, Windows mesh-loading fix, toolbox version check
- `.gitignore`
