
# _________________________________ Library _________________________________
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from collections import defaultdict
from tqdm import tqdm
import time
import doctest
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# _________________________________ SL NN class _________________________________
class SL_Network(nn.Module):
    def __init__(self, state_num, hidden_units_num):
        super(SL_Network, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num

        self.fc1 = nn.Linear(self.state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, 1)


    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        output = torch.sigmoid(self.fc2(h1))
        return output


# _________________________________ SL class _________________________________
class SupervisedLearning:
  def __init__(self, num_players, hidden_units_num, lr, epochs, sampling_num, kuhn_trainer_for_sl):

    self.NUM_PLAYERS = num_players
    self.STATE_BIN_LEN = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num

    self.card_rank  = kuhn_trainer_for_sl.card_rank


    self.sl_network = SL_Network(state_num=self.STATE_BIN_LEN, hidden_units_num=self.hidden_units_num)
    self.optimizer = optim.Adam(self.sl_network.parameters(), lr=self.lr)
    self.loss_fn = nn.BCELoss()



  def SL_learn(self, memory, target_player, update_strategy):

    #train
    self.sl_network.train()

    for _ in range(self.epochs):
      self.optimizer.zero_grad()

      samples = self.reservoir_sampling(memory, self.sampling_num)

      train_X = np.array([])
      train_y = np.array([])
      for one_s_a_set in samples:
        if one_s_a_set is not None:
          train_i = self.From_episode_to_bit([one_s_a_set])
          train_X = np.append(train_X, train_i[0])
          train_y = np.append(train_y, train_i[1])


      inputs = torch.from_numpy(train_X).float().reshape(-1,self.STATE_BIN_LEN)
      targets = torch.from_numpy(train_y).float().reshape(-1,1)

      outputs = self.sl_network.forward(inputs).reshape(-1,1)

      loss = self.loss_fn(outputs, targets)
      loss.backward()
      self.optimizer.step()



    # eval
    self.sl_network.eval()
    for node_X , _ in update_strategy.items():
      if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
        inputs_eval = torch.from_numpy(self.make_X(node_X)).float().reshape(-1,self.STATE_BIN_LEN)
        y = self.sl_network.forward(inputs_eval).detach().numpy()

        #tensor → numpy
        update_strategy[node_X] = np.array([1-y[0][0], y[0][0]])



  def whether_put_memory_i(self, i, d, k):
    if i < k:
      self.new_memory[i] = d
    else:
      r = random.randint(1, i)
      if r < k:
        self.new_memory[r] = d



  def reservoir_sampling(self, memory, k):
    self.new_memory = [None for _ in range(k)]
    for i in range(len(memory)):
      self.whether_put_memory_i(i, memory[i], k)

    return self.new_memory




  def From_episode_to_bit(self, one_s_a_set):
    """return list
    >>> SupervisedLearning(2, 2).From_episode_to_bit([('Q', 'b')])
    (array([0, 1, 0, 0, 0, 0, 0]), array([1]))
    """
    for X, y in one_s_a_set:
      y_bit = self.make_y(y)
      X_bit = self.make_X(X)

    return (X_bit, y_bit)


  def make_y(self, y):
    if y == "p":
      y_bit = np.array([0])
    else:
      y_bit = np.array([1])
    return y_bit


  def make_X(self, X):

    X_bit = np.array([0 for _ in range(self.STATE_BIN_LEN)])
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
