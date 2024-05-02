from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import sys

# This script makes the end-effector perform pick, pour, and place tasks
# Note that this script may not work for every arm as it was designed for the wx250
# Make sure to adjust commanded joint positions and poses as necessary
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py  # python3 bartender.py if using ROS Noetic'

def main():
    bot = InterbotixManipulatorXS("rx200", "arm", "gripper")

    if (bot.arm.group_info.num_joints < 5):
        print('This demo requires the robot to have at least 5 joints!')
        sys.exit()

    bot.arm.go_to_home_pose()
    bot.arm.set_single_joint_position("waist", -1)
    bot.gripper.open()
    bot.arm.set_single_joint_position("wrist_angle", -1)
    bot.arm.set_single_joint_position("elbow", 0.9)
    bot.arm.set_single_joint_position("shoulder", 0.1)
    bot.gripper.close()
    bot.arm.go_to_home_pose()
    bot.arm.set_single_joint_position("waist", 1.1)
    bot.arm.set_single_joint_position("wrist_angle", -1.43)
    bot.arm.set_single_joint_position("elbow", 0.56)
    bot.arm.set_single_joint_position("shoulder", 0.23)
    bot.gripper.open()
    bot.arm.go_to_home_pose()
    bot.gripper.close()
    bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()
    

