from torch import nn
import torch
import gymnasium as gym
import numpy as np
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import os
import time

# Same network architecture as training
class ConvolutionalNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            # Conv1: 32 filters, 8x8, stride 4
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Conv2: 64 filters, 4x4, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Conv3: 64 filters, 3x3, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            # FC1: 512 units
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            
            # Output: one per action
            nn.Linear(512, env.action_space.n)
        )
    
    def forward(self, x):
        # x shape: (batch, 4, 84, 84)
        x = self.conv(x)  # (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 3136)
        x = self.fc(x)  # (batch, num_actions)
        return x
    
    def act(self, obs, device, return_q_values=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        obs_t = obs_t.unsqueeze(0).to(device)  # Move to same device as model

        q_values = self(obs_t)

        action = q_values.argmax(dim=1)[0].item()
        
        if return_q_values:
            return action, q_values[0].cpu().detach().numpy()
        return action

# Hyperparameters (same as training)
NO_OP_MAX = 30

# Create environment with same preprocessing as training
# Create base environment with rendering
base_env = gym.make('ALE/Breakout-v5', frameskip=1, render_mode='human')

# Apply same preprocessing (this wraps base_env)
env = AtariPreprocessing(
    base_env,
    screen_size=84,
    frame_skip=4,
    grayscale_obs=True,
    scale_obs=True,
    noop_max=NO_OP_MAX
)

env = FrameStackObservation(env, stack_size=4)

# Create network
network = ConvolutionalNetwork(env)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = network.to(device)
network.eval()  # Set to evaluation mode

# Load checkpoint
checkpoint_path = 'saved_models/checkpoint_step_100000.pth'

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    print("Available checkpoints:")
    if os.path.exists('saved_models'):
        for f in os.listdir('saved_models'):
            if f.endswith('.pth'):
                print(f"  - {f}")
    exit(1)

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load the model state
network.load_state_dict(checkpoint['online_net_state_dict'])

print(f"Model loaded successfully!")
print(f"Training step: {checkpoint.get('step', 'unknown')}")
print(f"Average reward at checkpoint: {checkpoint.get('avg_reward', 'unknown')}")
print(f"Action space: {env.action_space}")
print(f"Action space size: {env.action_space.n}")
print("\nStarting to play... (Close window to stop)")

# Play the game
obs, info = env.reset()
total_reward = 0
episode_count = 0
step_count = 0
initial_lives = info.get('lives', None)  # Track initial lives
current_lives = initial_lives

print("Game started! You should see the game window.")
print("First few actions will be printed to verify it's working...\n")
print(f"Observation shape: {obs.shape}")
print(f"Observation dtype: {obs.dtype}")
print(f"Observation min/max: {obs.min()}/{obs.max()}")
if initial_lives is not None:
    print(f"Initial lives: {initial_lives}\n")
else:
    print("(Lives info not available)\n")

try:
    while True:
        # Get action from network (no exploration - pure exploitation)
        if step_count < 10:
            # For first 10 steps, get Q-values for debugging
            action, q_vals = network.act(obs, device, return_q_values=True)
        else:
            action = network.act(obs, device)
        
        # Fire once at the very start to get the ball moving
        # After that, use the agent's actual decisions
        if step_count == 0:
            action = 1  # Fire action to start the ball
        
        # Step environment (this should step the base_env through wrappers)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Check info for life information (Breakout has multiple lives)
        # In Breakout, episode ends when all lives (usually 5) are lost
        lives = info.get('lives', None)
        
        # If a life was lost (lives decreased), fire the new ball automatically
        if lives is not None and current_lives is not None:
            if lives < current_lives:
                print(f"\nLife lost! Lives: {current_lives} -> {lives}. Auto-firing new ball...")
                # Fire the new ball that appeared after losing a life
                action = 1  # Fire action
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                # Update lives after firing
                lives = info.get('lives', None)
                # Render after firing to see the ball launch
                base_env.render()
            current_lives = lives  # Update current lives tracking
        
        # Render from the base environment
        # Since base_env is wrapped inside env, stepping env should update base_env
        base_env.render()
        
        # Debug output for first 10 steps to see Q-values
        if step_count <= 10:
            print(f"Step {step_count}, Action: {action}, Q-values: {q_vals}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Lives: {lives}, Total: {total_reward:.2f}")
        elif step_count <= 30:
            print(f"Step {step_count}, Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Lives: {lives}, Total: {total_reward:.2f}")
        # Then every 100 steps
        elif step_count % 100 == 0:
            print(f"Step {step_count}, Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Lives: {lives}, Total: {total_reward:.2f}")
        
        # Small delay to see the game (increase if too fast)
        time.sleep(0.05)  # Increased delay to see better
        
        # Check if episode ended and restart automatically
        if done or truncated:
            episode_count += 1
            print(f"\n{'='*50}")
            print(f"Episode {episode_count} finished!")
            print(f"Episode reward: {total_reward:.2f}")
            print(f"Steps in episode: {step_count}")
            print(f"Average reward so far: {total_reward / episode_count:.2f}")
            print(f"{'='*50}\n")
            print("Restarting new episode in 1 second...\n")
            
            # Small delay before restarting
            time.sleep(1.0)
            
            # Reset environment for new episode
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
            current_lives = info.get('lives', None)  # Reset life tracking
            print("New episode started!\n")

except KeyboardInterrupt:
    print("\n\nStopped by user")
    env.close()

