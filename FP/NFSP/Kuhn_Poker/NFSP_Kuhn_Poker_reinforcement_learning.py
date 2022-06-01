
# _________________________________ Library _________________________________
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

# _________________________________ RL NN class _________________________________
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


# _________________________________ RL class _________________________________
class ReinforcementLearning:
  def __init__(self, num_players, hidden_units_num, lr, epochs, sampling_num, gamma, tau, kuhn_trainer_for_rl):
    self.NUM_PLAYERS = num_players
    self.num_actions = 2
    self.action_id = {"p":0, "b":1}
    self.STATE_BIT_LEN = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num
    self.gamma = gamma
    self.tau = tau
    self.card_rank  = kuhn_trainer_for_rl.card_rank


    self.deep_q_network = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)
    self.deep_q_network_target = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)


    for target_param, param in zip(self.deep_q_network_target.parameters(), self.deep_q_network.parameters()):
        target_param.data.copy_(param.data)


    self.optimizer = optim.SGD(self.deep_q_network.parameters(), lr=self.lr)


  def RL_learn(self, memory, target_player, update_strategy, k):

    self.deep_q_network.train()
    self.deep_q_network_target.eval()
    self.epsilon = 0.06/(k**0.5)
    self.update_frequency = 100

    # train
    if len(memory) < self.sampling_num:
      return

    for _ in range(self.epochs):
      self.optimizer.zero_grad()
      samples = random.sample(memory, self.sampling_num)

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

      train_states = torch.from_numpy(train_states).float().reshape(-1,self.STATE_BIT_LEN)
      train_actions = torch.from_numpy(train_actions).float().reshape(-1,1)
      train_rewards = torch.from_numpy(train_rewards).float().reshape(-1,1)
      train_next_states = torch.from_numpy(train_next_states).float().reshape(-1,self.STATE_BIT_LEN)
      train_done = torch.from_numpy(train_done).float().reshape(-1,1)

      outputs = self.deep_q_network_target(train_next_states).detach().max(axis=1)[0].unsqueeze(1)


      q_targets = train_rewards + (1 - train_done) * self.gamma * outputs

      q_now = self.deep_q_network(train_states)
      q_now = q_now.gather(1, train_actions.type(torch.int64))


      loss = F.mse_loss(q_targets, q_now)


      loss.backward()
      self.optimizer.step()


    if k % self.update_frequency ==  0:
      self.parameter_update()



    #eval
    self.deep_q_network.eval()
    for node_X , _ in update_strategy.items():
      if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
        inputs_eval = torch.from_numpy(self.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)
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
          self.tau * param.data + (1.0 - self.tau) * target_param.data)




  def make_action_bit(self, y):
    if y == "p":
      y_bit = np.array([0])
    else:
      y_bit = np.array([1])
    return y_bit


  def make_state_bit(self, X):

    X_bit = np.array([0 for _ in range(self.STATE_BIT_LEN)])

    if X != None:

      X_bit[self.card_rank[X[0]]] = 1

      for idx, Xi in enumerate(X[1:]):
        if Xi == "p":
          X_bit[(self.NUM_PLAYERS+1) + 2*idx] = 1
        else:
          X_bit[(self.NUM_PLAYERS+1) + 2*idx +1] = 1

    return X_bit


  def first_bit(self, X0, X_bit):
    if X0 == "J":
      X_bit[0] = 1
    elif X0 == "Q":
      X_bit[1] = 1
    elif X0 == "K":
      X_bit[2] = 1
    return X_bit








doctest.testmod()
