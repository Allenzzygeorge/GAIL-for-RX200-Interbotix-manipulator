import numpy as np
import pybullet as p
import pybullet_data
import torch
import time
from torch import nn

# Define joint limits and names as before
joint_limits = {
    'waist': (-3.14159, 3.14159), 
    'shoulder': (-1.88496, 1.97222),
    'elbow': (-1.88496, 1.62316),
    'wrist_angle': (-1.74533, 2.14675),
    'wrist_rotate': (-3.14159, 3.14159)
}
joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']
num_joints = len(joint_limits)

class Generator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Ensure this matches the number of joints
        )

    def forward(self, state):
        return self.model(state)

def load_robot(urdf_path):
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return robot_id

def apply_action(robot_id, action):
    # Apply joint commands directly without the gripper
    for i, joint_command in enumerate(action):
        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_command)

def get_observation(robot_id):
    joint_states = p.getJointStates(robot_id, range(num_joints))
    joint_positions = [state[0] for state in joint_states]
    return np.array(joint_positions)

# Initialize models with correct dimensions
state_dim = 4  # Number of input dimensions (e.g., number of joints)
action_dim = 4 

def normalize_state(state):
    # Normalize each joint angle to range [0, 1] based on its limits
    normalized = []
    for i, angle in enumerate(state):
        # Assuming joint_limits is a dictionary with joint names as keys and (min, max) tuples as values
        joint_name = joint_names[i]  # Ensure this matches the order used in get_observation
        min_val, max_val = joint_limits[joint_name]
        # Normalize current joint state
        normalized_angle = (angle - min_val) / (max_val - min_val)
        normalized.append(normalized_angle)
    return np.array(normalized)

def denormalize_action(action):
    # Denormalize the action to the expected joint control ranges
    denormalized = []
    for i, act in enumerate(action):
        joint_name = joint_names[i]  # Ensure this matches your model's output order
        min_val, max_val = joint_limits[joint_name]
        # Denormalize action
        joint_action = act * (max_val - min_val) + min_val
        denormalized.append(joint_action)
    return np.array(denormalized)


def main():
    urdf_path = '/home/saumya_rlc/Documents/rx200.urdf'
    robot_id = load_robot(urdf_path)
    model_path = '/home/saumya_rlc/Desktop/RLC_project/final_policy_dcg.pth'
    policy = Generator(state_dim, action_dim)  # Correct dimensions for input/output
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    for _ in range(1000):
        current_state = get_observation(robot_id)
        normalized_state = normalize_state(current_state)
        state_tensor = torch.tensor([normalized_state], dtype=torch.float32)
        with torch.no_grad():
            action = policy(state_tensor).numpy()[0]
        real_action = denormalize_action(action)
        apply_action(robot_id, real_action)
        p.stepSimulation()
        time.sleep(1./240.)


    print("Simulation complete. Press 'Enter' to exit.")
    input()
    p.disconnect()

if __name__ == "__main__":
    main()
