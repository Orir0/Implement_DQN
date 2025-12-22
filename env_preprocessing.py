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
NO_OP_MAX = 30 # max number of 'do nothing' actions to be performed by the agent at the start of an episode

def make_atari_breakout_enviroment_for_dqn():

    #basee enviroment
    env = gym.make('ALE/Breakout-v5', frameskip=1) # this version of breakout alread has frame skip = 4 by default so i set it to 1 for the preprocessing to handle

    #apply preprocessing - PRE PROCESSING START
    env  = AtariPreprocessing(
        env,
        screen_size=84,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=True,
        noop_max= NO_OP_MAX
    ) #this are the defualt for this function - just wrote them for clarity

    env = FrameStackObservation(env, stack_size=4)

    return env

    # PREPROCESSING FINISHED