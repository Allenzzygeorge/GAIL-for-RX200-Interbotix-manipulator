import numpy as np
import pybullet as p
import pybullet_data
import time

# ... [Your previous code defining the joint limits and other functions] ...
 #Define joint limits based on provided details
joint_limits = {
    'waist': (-3.14159, 3.14159), 
    'shoulder': (-1.88496, 1.97222),
    'elbow': (-1.88496, 1.62316),
    'wrist_angle': (-1.74533, 2.14675),
    'wrist_rotate': (-3.14159, 3.14159)
}
num_joints = len(joint_limits)

# Joint names in the order they are indexed in the simulation
joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate', 'gripper']


def load_robot(urdf_path):
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return robot_id

def perform_action_sequence(robot_id):
    home_pose = [0] * num_joints
    sleep_pose = home_pose  # Define the sleep pose if it's different from the home pose
    action_sequence = [
        ('go_to_home_pose', home_pose),
        ('waist', -1.6),
        ('open_gripper', 0.5),  # Open position
        ('wrist_angle', -1.15),
        ('elbow', 0.63),
        ('shoulder', 0.63),
        ('close_gripper', 0),  # Closed position
        ('go_to_home_pose', home_pose),
        ('waist', 1.6),
        ('wrist_angle', -1.53),
        ('elbow', 0.86),
        ('shoulder', 0.63),
        ('open_gripper', 0.5),
        ('go_to_home_pose', home_pose),
        ('close_gripper', 0),
        ('go_to_sleep_pose', sleep_pose),
    ]

    desired_velocity=0.2
    for action_name, target_position in action_sequence:
        if 'gripper' in action_name:  
            gripper_joint_index = 5  
            # Set the maximum force to apply to the joint (optional but can help with smoother movement)
            p.setJointMotorControl2(robot_id, gripper_joint_index, p.POSITION_CONTROL, targetPosition=target_position, maxVelocity=desired_velocity)
        elif 'pose' in action_name:  
            for i, joint_pos in enumerate(target_position):
                # Set the maximum force and velocity for smoother movement
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_pos, maxVelocity=desired_velocity)
        else:  
            joint_index = joint_names.index(action_name)
            # Set the maximum force and velocity for smoother movement
            p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_position, maxVelocity=desired_velocity)
        # Wait for the action to complete
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)

def main():
    urdf_path = '/home/saumya_rlc/Documents/rx200.urdf'
    robot_id = load_robot(urdf_path)

    # Perform the sequence of actions
    perform_action_sequence(robot_id)

    print("Sequence completed. Press 'Enter' to exit...")
    input()
    p.disconnect()

if __name__ == "__main__":
    main()
