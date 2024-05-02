import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Simulated joint states and actions for demonstration
# For example, each joint has a starting position of 0 radians
joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle']
num_joints = len(joint_names)
initial_state = np.zeros(num_joints)  # all joints start at 0 radians

# Simulate the expert commands
commands = [
    ("go_to_home_pose", np.zeros(num_joints)),  # Home pose, all joints to 0
    ("waist", -1.6),
    ("wrist_angle", -1.15),
    ("elbow", 0.63),
    ("shoulder", 0.63),
    ("waist", 1.6),
    ("wrist_angle", -1.53),
    ("elbow", 0.86),
    ("shoulder", 0.63),
    ("go_to_sleep_pose", initial_state)  # Sleep pose, might be different than home
]

# Generate state-action pairs
states = [initial_state]
actions = []

current_state = initial_state.copy()
for command in commands:
    action = np.zeros(num_joints)
    if command[0] == "go_to_home_pose" or command[0] == "go_to_sleep_pose":
        next_state = command[1]
        action = next_state - current_state
    else:
        joint_index = joint_names.index(command[0])
        next_state = current_state.copy()
        next_state[joint_index] = command[1]
        action[joint_index] = command[1] - current_state[joint_index]
    
    states.append(next_state)
    actions.append(action)
    current_state = next_state

states = np.array(states[:-1])  # Exclude the last state since there's no action from it
actions = np.array(actions[1:])

# Normalize data
scaler_states = MinMaxScaler()
scaler_actions = MinMaxScaler()
scaled_states = scaler_states.fit_transform(states)
scaled_actions = scaler_actions.fit_transform(actions)

# Convert to torch tensors
expert_states = torch.tensor(scaled_states[:-1], dtype=torch.float32)  # all but last
expert_actions = torch.tensor(scaled_actions, dtype=torch.float32)  # aligned with states

# Define models
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.model(state)

# Initialize models and optimizers
state_dim = expert_states.shape[1]
action_dim = expert_actions.shape[1]

#print("state dim".state_dim.shape(),"action dim", action_dim.shape())

policy = Generator(state_dim, action_dim)
discriminator = Discriminator(state_dim + action_dim)

optimizer_policy = torch.optim.Adam(policy.parameters(), lr=0.001)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)


discriminator_losses = []
policy_losses = []

def train_and_save_final_model(expert_states, expert_actions, policy, discriminator, epochs, optimizer_policy, optimizer_discriminator, save_path):
    criterion = nn.BCELoss()
    
    # Debug: Print shapes of states and actions
    print("States shape:", expert_states.shape)
    print("Actions shape:", expert_actions.shape)
    
    for epoch in range(epochs):
        # Generator generates actions
        gen_actions = policy(expert_states)

        # Ensure actions generated have the correct shape
        if gen_actions.shape != expert_actions.shape:
            print("Generated actions shape mismatch:", gen_actions.shape, "expected:", expert_actions.shape)
            continue  # Skip this epoch or handle error

        # Discriminator evaluates both real and generated actions
        real = discriminator(expert_states, expert_actions)
        fake = discriminator(expert_states, gen_actions.detach())
        d_loss = criterion(real, torch.ones_like(real)) + criterion(fake, torch.zeros_like(fake))
        
        # Update discriminator
        optimizer_discriminator.zero_grad()
        d_loss.backward()
        optimizer_discriminator.step()

        # Update generator
        policy_loss = -torch.log(discriminator(expert_states, gen_actions)).mean()
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        discriminator_losses.append(d_loss.item())
        policy_losses.append(policy_loss.item())
        
        
        print(f'Epoch {epoch}: Discriminator Loss: {d_loss.item()}, Policy Loss: {policy_loss.item()}')

    # Save models after training completes
    torch.save(policy.state_dict(), f'{save_path}/final_policy_dcg.pth')
    torch.save(discriminator.state_dict(), f'{save_path}/final_discriminator_dcg.pth')


    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()



# Define the directory to save the models
save_path = '/home/saumya_rlc/Desktop/RLC_project'


# Run training and save the final models
train_and_save_final_model(
    expert_states,
    expert_actions,
    policy,
    discriminator,
    epochs=100000,
    optimizer_policy=optimizer_policy,
    optimizer_discriminator=optimizer_discriminator,
    save_path=save_path
)
