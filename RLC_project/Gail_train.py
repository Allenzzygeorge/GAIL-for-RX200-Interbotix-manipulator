import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load data
data_path = '/home/saumya_rlc/Downloads/data1.csv'
df = pd.read_csv(data_path)
print(df.shape)  # Confirm the shape
print(df.columns)  # Confirm column names

# Create a joint identifier and pivot the DataFrame
df['joint'] = 'Joint_' + df['id'].astype(str)
df_pivot = df.pivot_table(index='marker_time', columns='joint', values=['x', 'y', 'z'], aggfunc='first')

# Flatten the columns after pivoting
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
df_pivot.reset_index(inplace=True)

# Optionally, drop 'marker_time' if it's not needed as a feature
positions = df_pivot.drop('marker_time', axis=1).values

scaler = MinMaxScaler()
scaled_positions = scaler.fit_transform(positions)
print("Scaled positions shape:", scaled_positions.shape)  # Check the new shape

# Create state-action pairs
X = scaled_positions[:-1]  # states
y = scaled_positions[1:] - scaled_positions[:-1]  # actions

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
state_dim = X.shape[1]
action_dim = y.shape[1]

policy = Generator(state_dim, action_dim)
discriminator = Discriminator(state_dim + action_dim)

optimizer_policy = torch.optim.Adam(policy.parameters(), lr=0.001)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Convert data to torch tensors
expert_states = torch.tensor(X, dtype=torch.float32)
expert_actions = torch.tensor(y, dtype=torch.float32)

discriminator_losses = []
policy_losses = []

def train_and_save_final_model(expert_states, expert_actions, policy, discriminator, epochs, optimizer_policy, optimizer_discriminator, save_path):
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        # Generator generates actions
        gen_actions = policy(expert_states)

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
    torch.save(policy.state_dict(), f'{save_path}/final_policy.pth')
    torch.save(discriminator.state_dict(), f'{save_path}/final_discriminator.pth')

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
    epochs=200000,
    optimizer_policy=optimizer_policy,
    optimizer_discriminator=optimizer_discriminator,
    save_path=save_path
)

