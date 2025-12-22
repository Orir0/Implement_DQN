from torch import nn
import torch
import gymnasium as gym
from collections import deque
import itertools
import random
import numpy as np
import ale_py 
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import os

#creating the neural network
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
        
        # Calculate flattened size after conv layers
        # Input: 84x84x4
        # After Conv1 (stride 4): ~20x20x32
        # After Conv2 (stride 2): ~9x9x64
        # After Conv3 (stride 1): ~7x7x64 = 3136
        
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
    
    def act(self, obs, device):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        obs_t = obs_t.unsqueeze(0).to(device)  # Move to same device as model

        q_values = self(obs_t)

        action = q_values.argmax(dim=1)[0].item()

        return action