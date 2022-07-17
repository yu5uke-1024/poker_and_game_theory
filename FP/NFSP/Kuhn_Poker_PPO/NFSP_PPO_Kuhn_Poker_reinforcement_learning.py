
# _________________________________ Library _________________________________
from platform import node
from unittest.mock import seal
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

    return action_log_probability, torch.squeeze(state_value), distribution_entropy


class PPO:
  def __init__(self, num_players, hidden_units_num, sampling_num):
    self.NUM_PLAYERS = num_players
    self.num_actions = 2
    self.num_states = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.sampling_num = sampling_num
    self.gamma = 0.1
    self.epochs = 1
    self.eps_clip = 0.2


    self.current_policy = ActorCriticNetwork(self.num_states, self.num_actions, self.hidden_units_num)
    self.optimizer = optim.Adam(self.current_policy.parameters())

    self.old_policy = ActorCriticNetwork(self.num_states, self.num_actions, self.hidden_units_num)
    self.old_policy.load_state_dict(self.current_policy.state_dict())

    self.loss_function = nn.MSELoss()



  def select_action(self, node_X):
    with torch.no_grad():
      action, action_log_prob = self.old_policy.make_action(node_X)

      return action, action_log_prob, 1


  def RL_learn(self, memory, target_player):
    # memory[0] → s, a, r, s_prime, done

    rewards = []
    discounted_reward = 0


    for reward, done in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
      if done:
        discounted_reward = 0
      discounted_reward = reward + (self.gamma * discounted_reward)
      rewards.append(discounted_reward)

    rewards.reverse()
    rewards = torch.tensor(rewards)



    old_states = torch.tensor(memory.states).float().reshape(-1,self.num_states).detach()
    old_actions = torch.tensor(memory.actions).float().reshape(-1,1).detach()
    old_logprobs = torch.tensor(memory.logprobs).float().reshape(-1,1).detach()

    for _ in range(self.epochs):
      logprobs, state_values, distribution_entropy = self.current_policy.evaluate_action(old_states, old_actions)

      # importance ratio r(θ)
      ratios = torch.exp(logprobs - old_logprobs.detach())

      # adavantage
      advantages = rewards - state_values.detach()

      #clip した方と してない方 比べて小さい方を選択
      loss_unclipped = ratios * advantages
      loss_clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

      actor_loss = - torch.min(loss_unclipped, loss_clipped)

      # Critic loss: critic loss - entropy
      critic_loss = 0.5 * self.loss_function(rewards, state_values) - 0.01 * distribution_entropy


      loss = actor_loss + critic_loss

      self.optimizer.zero_grad()
      loss.mean().backward()
      self.optimizer.step()


    self.old_policy.load_state_dict(self.current_policy.state_dict())


class RL_memory:
  def __init__(self):
      self.states = []
      self.actions = []
      self.rewards = []
      self.is_terminals = []
      self.logprobs = []

  def del_memory(self):
      self.states = []
      self.actions = []
      self.rewards = []
      self.is_terminals = []
      self.logprobs = []

  def display(self):
    return self.states, self.actions, self.rewards, self.logprobs



an = ActorCriticNetwork(state_num=7, action_num=2, hidden_units_num=32)
data = torch.Tensor(np.random.rand(10,7))
#print(an.make_action(data, "1"))



class PPO_RL_memory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, memory):
        self.states.append(memory["observation"])
        self.actions.append(memory["action"])
        self.log_probs.append(memory["log_prob"])
        self.values.append(memory["value"])
        self.rewards.append(memory["reward"])
        self.dones.append(memory["done"])

    def delete(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


    def get_size(self):
        return len(self.states)


    def get_batch(self):
        #dataをbatch_sizeに切り分ける そのindexを作る
        size = self.get_size()
        batch_start_index = np.arange(0, size, self.batch_size)
        state_index = np.arange(size, dtype=np.int64)
        np.random.shuffle(state_index)
        batch_index = [state_index[i:i+self.batch_size] for i in batch_start_index]


        #npだと array[batch] で 取得可能
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
            np.array(self.values), np.array(self.rewards), np.array(self.dones), batch_index
