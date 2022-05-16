#ライブラリ
from cmath import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from collections import defaultdict
import sys
from tqdm import tqdm
import time
import doctest
import copy
from sklearn.neural_network import MLPClassifier
from collections import deque
import math

import FSP_Kuhn_Poker_trainer



class ReinforcementLearning:
  def __init__(self):
    self.gamma = 1
    self.NUM_ACTIONS = 2
    self.player_0_state = {"J":0, "Q":1, "K":2, "Jpb":3, "Qpb":4, "Kpb":5}
    self.player_1_state = {"Jp":0, "Jb":1, "Qp":2, "Qb":3, "Kp":4, "Kb":5}
    self.action_id = {"p":0, "b":1}



  def RL_train(self, memory, target_player, update_strategy, q_value, k, rl_algo):
    self.alpha = 0.05/ (1+0.003*(k**0.5))
    self.T = 1 / (1+ 0.02*(k**0.5))
    self.epsilon = 0.6/((k+1)**0.5)

    if target_player == 0:
      for memory_i in memory:
        if len(memory_i) == 4:
          s = self.player_0_state[memory_i[0]]
          a = self.action_id[memory_i[2]]
          r = FSP_Kuhn_Poker_trainer.KuhnTrainer().Return_payoff_for_terminal_states(memory_i, target_player)
          q_value[s][a] = q_value[s][a]  + self.alpha*(r - q_value[s][a])

        else:
          s = self.player_0_state[memory_i[0]]
          a = self.action_id[memory_i[2]]
          r = 0
          s_prime = self.player_0_state[memory_i[0]+memory_i[2:4]]
          q_value[s][a] = q_value[s][a]  + self.alpha*(r + self.gamma*max(q_value[s_prime]) - q_value[s][a])

          s = self.player_0_state[memory_i[0]+memory_i[2:4]]
          a = self.action_id[memory_i[4]]
          r = FSP_Kuhn_Poker_trainer.KuhnTrainer().Return_payoff_for_terminal_states(memory_i, target_player)
          q_value[s][a] = q_value[s][a]  + self.alpha*(r - q_value[s][a])


      if rl_algo == "boltzmann":
        q_value_boltzmann = np.zeros((6,2))
        for si in range(6):
          for ai in range(2):
            q_value_boltzmann[si][ai] = math.exp(q_value[si][ai]/self.T)

        for state, idx in self.player_0_state.items():
          update_strategy[state] = q_value_boltzmann[self.player_0_state[state]] / sum(q_value_boltzmann[self.player_0_state[state]])

      elif rl_algo == "epsilon-greedy":
        for state, idx in self.player_0_state.items():
          if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
            action = np.random.randint(2)
            if action == 0:
              update_strategy[state] = np.array([1, 0], dtype=float)
            else:
              update_strategy[state] = np.array([0, 1], dtype=float)

          else:
            if q_value[self.player_0_state[state]][0] > q_value[self.player_0_state[state]][1]:
              update_strategy[state] = np.array([1, 0], dtype=float)
            else:
              update_strategy[state] = np.array([0, 1], dtype=float)


    elif target_player == 1:
      for memory_i in memory:
          s = self.player_1_state[memory_i[1:3]]
          a = self.action_id[memory_i[3]]
          r = FSP_Kuhn_Poker_trainer.KuhnTrainer().Return_payoff_for_terminal_states(memory_i, target_player)
          q_value[s][a] = q_value[s][a]  + self.alpha*(r - q_value[s][a])


      if rl_algo == "boltzmann":
        q_value_boltzmann = np.zeros((6,2))
        for si in range(6):
          for ai in range(2):
            q_value_boltzmann[si][ai] = math.exp(q_value[si][ai]/self.T)

        for state, idx in self.player_1_state.items():
          update_strategy[state] = q_value_boltzmann[self.player_1_state[state]] / sum(q_value_boltzmann[self.player_1_state[state]])


      elif rl_algo == "epsilon-greedy":
        for state, idx in self.player_1_state.items():
          if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
            action = np.random.randint(2)
            if action == 0:
              update_strategy[state] = np.array([1, 0], dtype=float)
            else:
              update_strategy[state] = np.array([0, 1], dtype=float)

          else:
            if q_value[self.player_1_state[state]][0] > q_value[self.player_1_state[state]][1]:
              update_strategy[state] = np.array([1, 0], dtype=float)
            else:
              update_strategy[state] = np.array([0, 1], dtype=float)
