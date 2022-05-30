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
    self.max_len_X_bit = (self.num_players + 1) + 2*(self.num_players *2 - 2)


  def RL_train_DQN(self, memory, target_player, strategy):
    pass



class DQN(nn.Module):

    def __init__(self, state_num, action_num, hidde_units_num):
        super(DQN, self).__init__()
        self.hidden_units_num = hidde_units_num
        self.fc1 = nn.Linear(state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, action_num)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        output = self.fc2(h1)
        return output



doctest.testmod()
