# Plants_Do_be_Watered
Over engineered Watering Can

Take plant from shelf then place on desk and use another robot arm to water it 

Project description
ShelfCare automates watering plants stored on a wall shelf beyond easy reach. Arm 1 lifts a potted plant from the shelf and places it on a platform. Arm 2, the reBot B601-DM, then picks up a watering can and pours a measured amount into the pot. Once watering is complete, Arm 1 returns the plant to its shelf position. A simulated moisture sensor decides which plants need attention. Safety features include a keep-out zone around the platform during watering, collision detection along both arms' paths, and a physical e-stop on the reBot. A Python GUI lets an operator jog either arm and monitor system state.
Real robot
 reBot B601-DM — used for the watering-can task, since its six-axis reach and gripper payload suit lifting and tilting a filled can accurately.
Each student's robot model

The Stäubli TX2-60, a six-DoF industrial arm absent from both the 41013 catalogue and Corke's toolbox. Its published kinematic dimensions and 5 kg payload comfortably support lifting a potted plant off a shelf and placing it on the watering platform without tipping. 
