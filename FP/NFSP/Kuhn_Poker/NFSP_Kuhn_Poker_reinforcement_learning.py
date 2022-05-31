#ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import doctest
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import NFSP_Kuhn_Poker_trainer



class ReinforcementLearning:
  def __init__(self, infoSet_dict_player, num_players, num_actions):
    self.gamma = 1
    self.num_players = num_players
    self.num_actions = num_actions
    self.action_id = {"p":0, "b":1}
    self.kuhn_trainer = NFSP_Kuhn_Poker_trainer.KuhnTrainer(num_players=self.num_players)
    self.num_states = (self.num_players + 1) + 2*(self.num_players *2 - 2)

    self.card_order  = self.make_card_order(self.num_players)

    self.config_rl = dict(
      hidden_units_num= 64,
      lr = 0.1,
      epochs = 10,
      sampling_num = 30,
      gamma = 1.0,
      tau = 0.1
    )

    self.deep_q_network = DQN(state_num = self.num_states, action_num = self.num_actions, hidden_units_num = self.config_rl["hidden_units_num"])
    self.deep_q_network_target = DQN(state_num = self.num_states, action_num = self.num_actions, hidden_units_num = self.config_rl["hidden_units_num"])


    for target_param, param in zip(self.deep_q_network_target.parameters(), self.deep_q_network.parameters()):
        target_param.data.copy_(param.data)



    self.optimizer = optim.SGD(self.deep_q_network.parameters(), lr=self.config_rl["lr"])


  def RL_learn(self, memory, target_player, update_strategy, k):

    self.deep_q_network.train()
    self.deep_q_network_target.eval()
    self.epsilon = 0.06/(k**0.5)

    # train
    if len(memory) < self.config_rl["sampling_num"]:
      return

    for _ in range(self.config_rl["epochs"]):
      self.optimizer.zero_grad()
      samples = random.sample(memory, self.config_rl["sampling_num"])

      train_states = np.array([])
      train_actions = np.array([])
      train_rewards = np.array([])
      train_next_states = np.array([])
      train_done = np.array([])

      for s, a, r, s_prime in samples:
        s_bit = self.make_state_bit(s)
        a_bit = self.make_action_bit(a)
        s_prime_bit = self.make_state_bit(s_prime)
        if s_prime == None:
          done = 1
        else:
          done = 0

        train_states = np.append(train_states, s_bit)
        train_actions = np.append(train_actions, a_bit)
        train_rewards = np.append(train_rewards, r)
        train_next_states = np.append(train_next_states, s_prime_bit)
        train_done = np.append(train_done, done)

      train_states = torch.from_numpy(train_states).float().reshape(-1,self.num_states)
      train_actions = torch.from_numpy(train_actions).float().reshape(-1,1)
      train_rewards = torch.from_numpy(train_rewards).float().reshape(-1,1)
      train_next_states = torch.from_numpy(train_next_states).float().reshape(-1,self.num_states)
      train_done = torch.from_numpy(train_done).float().reshape(-1,1)

      outputs = self.deep_q_network_target(train_next_states).detach().max(axis=1)[0].unsqueeze(1)


      q_targets = train_rewards + (1 - train_done) * self.config_rl["gamma"] * outputs

      q_now = self.deep_q_network(train_states)
      q_now = q_now.gather(1, train_actions.type(torch.int64))


      loss = F.mse_loss(q_targets, q_now)


      loss.backward()
      self.optimizer.step()


      self.parameter_update()


    #eval
    self.deep_q_network.eval()
    for node_X , _ in update_strategy.items():
      if (len(node_X)-1) % self.num_players == target_player :
        inputs_eval = torch.from_numpy(self.make_state_bit(node_X)).float().reshape(-1,self.num_states)
        y = self.deep_q_network.forward(inputs_eval).detach().numpy()

        if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
          action = np.random.randint(self.num_actions)
          if action == 0:
            update_strategy[node_X] = np.array([1, 0], dtype=float)
          else:
            update_strategy[node_X] = np.array([0, 1], dtype=float)

        else:
          if y[0][0] > y[0][1]:
            update_strategy[node_X] = np.array([1, 0], dtype=float)
          else:
            update_strategy[node_X] = np.array([0, 1], dtype=float)





  def parameter_update(self):
    for target_param, param in zip(self.deep_q_network_target.parameters(), self.deep_q_network.parameters()):
      target_param.data.copy_(
          self.config_rl["tau"] * param.data + (1.0 - self.config_rl["tau"]) * target_param.data)



  def make_card_order(self, num_players):
    """return dict
    >>> SupervisedLearning().make_card_order(2) == {'J':0, 'Q':1, 'K':2}
    True
    """
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    card_order = {}
    for i in range(num_players+1):
      card_order[card[11-num_players+i]] =  i

    return card_order


  def make_action_bit(self, y):
    if y == "p":
      y_bit = np.array([0])
    else:
      y_bit = np.array([1])
    return y_bit


  def make_state_bit(self, X):

    X_bit = np.array([0 for _ in range(self.num_states)])

    if X != None:

      X_bit[self.card_order[X[0]]] = 1

      for idx, Xi in enumerate(X[1:]):
        if Xi == "p":
          X_bit[(self.num_players+1) + 2*idx] = 1
        else:
          X_bit[(self.num_players+1) + 2*idx +1] = 1

    return X_bit


  def first_bit(self, X0, X_bit):
    if X0 == "J":
      X_bit[0] = 1
    elif X0 == "Q":
      X_bit[1] = 1
    elif X0 == "K":
      X_bit[2] = 1
    return X_bit




class DQN(nn.Module):

    def __init__(self, state_num, action_num, hidden_units_num):
        super(DQN, self).__init__()
        self.hidden_units_num = hidden_units_num
        self.fc1 = nn.Linear(state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, action_num)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        output = self.fc2(h1)
        return output



doctest.testmod()
