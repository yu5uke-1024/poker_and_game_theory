
# _________________________________ Library _________________________________
from platform import node
from importlib_metadata import distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import doctest
from collections import deque
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical



import warnings
warnings.filterwarnings('ignore')
doctest.testmod()


# PPO
class ActorCriticNetwork(nn.Module):
  def __init__(self, state_num, action_num, hidden_units_num):
    super(ActorCriticNetwork, self).__init__()
    self.hidden_units_num = hidden_units_num
    self.fc1 = nn.Linear(state_num, self.hidden_units_num)
    self.fc2 = nn.Linear(self.hidden_units_num, action_num)
    self.fc3 = nn.Linear(self.hidden_units_num, 1)

    self.softmax = nn.Softmax()

  def actor_forward(self, x):
    h1 = F.leaky_relu(self.fc1(x))
    output = self.softmax(self.fc2(h1))
    return output

  def critic_forward(self, x):
    h1 = F.leaky_relu(self.fc1(x))
    output = self.fc3(h1)
    return output


  def make_action(self, node_X):
    action_probability = self.actor_forward(node_X)


    action_distribution = Categorical(action_probability)
    # 確率分布に従って、行動をサンプリング
    action = action_distribution.sample()
    # probability log

    action_log_probability = action_distribution.log_prob(action)

    return action, action_log_probability


  def evaluate_action(self, node_X, action):
    state_value = self.critic_forward(node_X)
    action_probability = self.actor_forward(node_X)
    action_distribution = Categorical(action_probability)
    action_log_probability = action_distribution.log_prob(action)
    distribution_entropy = action_distribution.entropy()

    return action_probability, torch.squeeze(state_value), distribution_entropy


class PPO:
  def __init__(self, num_players, hidden_units_num, sampling_num):
    self.NUM_PLAYERS = num_players
    self.num_actions = 2
    self.num_states = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.sampling_num = sampling_num


    self.current_policy = ActorCriticNetwork(self.num_states, self.num_actions, self.hidden_units_num)
    self.optimizer = optim.Adam(self.current_policy.parameters())

    self.old_policy = ActorCriticNetwork(self.num_states, self.num_actions, self.hidden_units_num)
    self.old_policy.load_state_dict(self.current_policy.state_dict())

    self.loss_function = nn.MSELoss()

    self.memory = RL_memory()


  def select_action(self, node_X):
    with torch.no_grad():

      action, action_log_prob = self.old_policy.make_action(node_X)

      return action


  def RL_learn(self, memory):
    pass


class RL_memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.log_probs = []
    self.rewards = []
    self.done = []




an = ActorCriticNetwork(state_num=7, action_num=2, hidden_units_num=32)
data = torch.Tensor(np.random.rand(10,7))
#print(an.make_action(data, "1"))
