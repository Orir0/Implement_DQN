from pydoc import render_doc
from torch import nn
import torch
import gym
from collections import deque
import itertools
import random
import numpy as np

GAMMA = 0.95 # discount factor
BATCH_SIZE = 32 
BUFFER_SIZE = 50000 
MIN_REPLAY_SIZE = 10000 # the number of trasition we want in our replay buffer before starting to computre gradients
EPSILON_START = 1.0 
EPSILON_END = 0.02 
EPSILON_DECAY = 10000 
TARGET_UPDATE_FREQ = 10000 


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape)) # size of our input/state vector - how many nuerons in our input 

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)) # Q learning can only be used for discrit/finite action space
        

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype= torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

env = gym.make('CartPole-v0')

replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)
epsisod_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr = 5e-4)

#init replay buffer
obs, _ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, rew, done, truncated, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs, _ = env.reset()


# main training loop
obs, _ = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])


    rnd_sample  = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)
        new_obs, rew, done, truncated, _ = env.step(action)
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
        obs = new_obs
        epsisod_reward += rew

        if done:
            obs, _ = env.reset()
        
            rew_buffer.append(epsisod_reward)
            epsisod_reward = 0.0 

    # after solved, watch it play
    if len(rew_buffer) >= 100 :
        if np.mean(rew_buffer) >= 195:
            render_env = gym.make('CartPole-v0', render_mode = 'human')
            obs_render, _ = render_env.reset()

            while True:
                action  = online_net.act(obs_render)
                obs_render, _, done, _, _ = render_env.step(action) 
                render_env.render()
                if done:
                    obs_render, _ =  render_env.reset()

    #start gradient step 
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.array([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses,dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype= torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    #compute target
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim = 1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1- dones_t) * max_target_q_values

    #compute loss
    q_values = online_net(obses_t)

    action_q_values = torch.gather(input = q_values, dim = 1, index = actions_t)

    loss  = nn.functional.smooth_l1_loss(action_q_values, targets)

    #gradient descent step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #update network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    #logging 
    if step % 1000 == 0:
        print()
        print('step', step)
        print(f'Avg Reward {np.mean(rew_buffer)}')









