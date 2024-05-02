import numpy as np
import pybullet as p
import pybullet_data
import torch
from gym import spaces
import time
from Gail_train import Generator  # Import the policy class

# Assuming you have 5 joints: waist, shoulder, elbow, wrist_angle, wrist_rotate
num_joints = 5

# Define joint limits based on provided details
joint_limits = {
    'waist': (-180, 180),
    'shoulder': (-108, 113),
    'elbow': (-108, 93),
    'wrist_angle': (-100, 123),
    'wrist_rotate': (-180, 180)
}

# Create action space with normalized limits [-1, 1]
action_space = spaces.Box(low=-1, high=1, shape=(num_joints,), dtype=np.float32)

observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * num_joints,), dtype=np.float32)

def load_robot(urdf_path):
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return robot_id

def apply_action(robot_id, action):
    # Calculate the actual joint commands based on joint limits
    joint_commands = []
    for i, joint_name in enumerate(joint_limits):
        joint_range = joint_limits[joint_name]
        joint_command = joint_range[0] + (action[i] + 1) * 0.5 * (joint_range[1] - joint_range[0])
        joint_commands.append(joint_command)

    # Apply joint commands
    for j, joint_command in enumerate(joint_commands):
        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                jointIndex=j,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_command)

def get_observation(robot_id):
    joint_states = p.getJointStates(robot_id, range(num_joints))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    return np.array(joint_positions + joint_velocities)

def main():
    urdf_path = '/home/saumya_rlc/Documents/rx200.urdf'
    robot_id = load_robot(urdf_path)

    # Load the trained policy model
    state_dim = 4 #2 * num_joints  # 2 for each joint (position and velocity)
    action_dim = 4 #num_joints
    policy = Generator(state_dim, action_dim)
    policy.load_state_dict(torch.load('/home/saumya_rlc/Desktop/RLC_project/final_policy.pth'))
    policy.eval()

    for _ in range(1000):
        # Get the current state of the robot
        current_observation = get_observation(robot_id)
        state_tensor = torch.from_numpy(current_observation).float()

        # Use the trained policy to determine the next action
        with torch.no_grad():
            action = policy(state_tensor).numpy()

        # Apply the action to the environment
        apply_action(robot_id, action)

        # Step the simulation forward
        p.stepSimulation()
        time.sleep(1./240.)

    print("Press 'Enter' to exit...")
    input()
    p.disconnect()

if __name__ == "__main__":
    main()
