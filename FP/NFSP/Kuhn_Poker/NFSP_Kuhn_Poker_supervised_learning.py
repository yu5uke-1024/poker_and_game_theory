
# _________________________________ Library _________________________________
from turtle import update
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import itertools
from collections import defaultdict
from tqdm import tqdm
import time
import doctest
import copy
import wandb
from collections import deque


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from NFSP_Kuhn_Poker_trainer import KuhnTrainer

# _________________________________ SL NN class _________________________________
class SL_Network(nn.Module):
    def __init__(self, state_num, hidden_units_num):
        super(SL_Network, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num

        self.fc1 = nn.Linear(self.state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, 1)
        #self.fc3 = nn.Linear(self.state_num, 1)

        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.leaky_relu(self.fc1(x))

        #output = self.fc2(h1)
        #h2 = self.dropout(h1)

        output = torch.sigmoid(self.fc2(h1))


        return output


# _________________________________ SL class _________________________________
class SupervisedLearning:
  def __init__(self,train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, loss_function, kuhn_trainer_for_sl):

    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 2
    self.STATE_BIT_LEN = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num

    self.kuhn_trainer = kuhn_trainer_for_sl

    self.card_rank  = self.kuhn_trainer.card_rank


    self.sl_network = SL_Network(state_num=self.STATE_BIT_LEN, hidden_units_num=self.hidden_units_num)

    #self.optimizer = optim.Adam(self.sl_network.parameters(), lr=self.lr, weight_decay=5*(10**(-4)))
    self.optimizer = optim.Adam(self.sl_network.parameters(), lr=self.lr)


    self.loss_fn = loss_function
    #self.loss_fn = nn.CrossEntropyLoss()


  def SL_learn(self, memory, target_player, update_strategy, iteration_t):

    #train
    self.sl_network.train()

    #if len(memory) < self.sampling_num:
    #  return

    total_loss = 0


    for _ in range(self.epochs):

      samples = self.reservoir_sampling(memory, min(self.sampling_num, len(memory)))
      random.shuffle(samples)


      train_X = np.array([])
      train_y = np.array([])

      for one_s_a_set in samples:
        if one_s_a_set is not None:
          train_i = self.kuhn_trainer.from_episode_to_bit([one_s_a_set])
          train_X = np.append(train_X, train_i[0])
          train_y = np.append(train_y, train_i[1])

          #print(one_s_a_set, train_i)

      #print("")
      #print(samples)
      inputs = torch.from_numpy(train_X).float().reshape(-1,self.STATE_BIT_LEN)
      targets = torch.from_numpy(train_y).float().reshape(-1, 1)

      outputs = self.sl_network.forward(inputs)


      #loss = - (targets * outputs).sum(dim=-1).mean()
      loss = self.loss_fn(outputs, targets)

      #print(outputs, targets, loss)


      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      total_loss += loss.item()


    if iteration_t in [int(j) for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*4 , endpoint=False)] :
      if self.kuhn_trainer.wandb_save:
        wandb.log({'iteration': iteration_t, 'loss_sl': total_loss/self.epochs})


        #for node_X , _ in update_strategy.items():
          #if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
            #print(node_X, update_strategy[node_X])







    # eval
    self.sl_network.eval()
    with torch.no_grad():
      for node_X , _ in update_strategy.items():
        if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
          inputs_eval = torch.from_numpy(self.kuhn_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)

          y = self.sl_network.forward(inputs_eval).detach().numpy()[0]

          update_strategy[node_X] = np.array([1.0-y[0], y[0]])



            #update_strategy[node_X] = np.array(np.exp(y))



  def whether_put_memory_i(self, i, data, k):
    if i < k:
      self.new_memory[i] = data
    else:
      r = random.randint(1, i)
      if r < k:
        self.new_memory[r] = data



  def reservoir_sampling(self, memory, k):
    self.new_memory = [None for _ in range(k)]
    for i in range(len(memory)):
      self.whether_put_memory_i(i, memory[i], k)

    return self.new_memory


  def SL_train_AVG(self, memory, target_player, strategy, n_count):
    for one_s_a_set in memory:
      for X, y in [one_s_a_set]:
        if (len(X)-1) % self.NUM_PLAYERS == target_player :
          if y == "p":
            n_count[X] += np.array([1.0, 0.0], dtype=float)
          else:
            n_count[X] += np.array([0.0, 1.0], dtype=float)

    for node_X , action_prob in n_count.items():
        strategy[node_X] = n_count[node_X] / np.sum(action_prob)


    return strategy


doctest.testmod()
