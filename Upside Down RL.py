import time
from copy import deepcopy
import gym
import numpy as np
import torch
import torch.nn.functional as F

env = gym.make('LunarLander-v2')
