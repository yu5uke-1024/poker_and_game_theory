#ライブラリ
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

import FSP_Kuhn_Poker_trainer


class GenerateData:
  def __init__(self):
    pass

  def generate_data(self, pi_strategy, beta_strategy, n, m, eta):
    sigma_strategy = {}
    for infoset in FSP_Kuhn_Poker_trainer.KuhnTrainer().infoSets_dict:
      sigma_strategy[infoset] = (1-eta)*pi_strategy[infoset] + eta*beta_strategy[infoset]

    sigma_strategy_player0, sigma_strategy_player1 = self.strategy_split_player(sigma_strategy)
    beta_strategy_player0, beta_strategy_player1 = self.strategy_split_player(beta_strategy)
    D_history = []

    for ni in range(n):
      ni_episode = self.one_episode("", self.strategy_uion(sigma_strategy_player0, sigma_strategy_player1))
      D_history.append(ni_episode)

    for player_i in range(FSP_Kuhn_Poker_trainer.KuhnTrainer().NUM_PLAYERS):
      if player_i == 0:
        D_history_0 = []
        for mi in range(m):
          mi_episode = self.one_episode("", self.strategy_uion(beta_strategy_player0, sigma_strategy_player1))
          D_history_0.append(mi_episode)
        D_history_0 = D_history + D_history_0
      else:
        D_history_1 = []
        for mi in range(m):
          mi_episode = self.one_episode("", self.strategy_uion(beta_strategy_player1, sigma_strategy_player0))
          D_history_1.append(mi_episode)
        D_history_1 = D_history + D_history_1
    return [D_history_0, D_history_1]


  def strategy_split_player(self, strategy):
    strategy_player0 = {}
    strategy_player1 = {}
    for infoset, avg_strategy in strategy.items():
      if len(infoset) %2 != 0:
        strategy_player0[infoset] = avg_strategy
      else:
        strategy_player1[infoset] = avg_strategy

    return strategy_player0, strategy_player1


  def strategy_uion(self, strategy_player0, strategy_player1):
    union_strategy = {}
    for i, j in strategy_player0.items():
      union_strategy[i] = j
    for i, j in strategy_player1.items():
      union_strategy[i] = j
    return union_strategy


  def one_episode(self, history, strategy):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if FSP_Kuhn_Poker_trainer.KuhnTrainer().whether_terminal_states(history):
      return history

    elif FSP_Kuhn_Poker_trainer.KuhnTrainer().whether_chance_node(history):
      cards = np.array(["J", "Q", "K"])
      random.shuffle(cards)
      nextHistory = cards[0] + cards[1]
      return self.one_episode(nextHistory, strategy)

    infoSet = history[player] + history[2:]

    sampling_action = np.random.choice(list(range(FSP_Kuhn_Poker_trainer.KuhnTrainer().NUM_ACTIONS)), p=strategy[infoSet])
    nextHistory = history + ("p" if sampling_action == 0 else "b")
    return self.one_episode(nextHistory, strategy)
