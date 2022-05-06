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

import wandb
#wandb.init(project="FSP_project", name="fsp_dfs")
wandb.init(project="FSP_project", name="fsp_rl")

import FSP_Kuhn_Poker_supervised_learning
import FSP_Kuhn_Poker_reinforcement_learning
import FSP_Kuhn_Poker_generate_data



class KuhnTrainer:
  def __init__(self, train_iterations=10**4):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = 2
    self.PASS = 0
    self.BET = 1
    self.NUM_ACTIONS = 2
    self.avg_strategy = {}
    self.rank = {"J":1, "Q":2, "K":3}


  #return util for terminal state
  def Return_payoff_for_terminal_states(self, history, target_player_i):
    """return int
    >>> KuhnTrainer().Return_payoff_for_terminal_states("JKpp", 0)
    -1
    >>> KuhnTrainer().Return_payoff_for_terminal_states("QKpp", 1)
    1
    """
    plays = len(history)
    player = plays % 2
    opponent = 1 - player
    terminal_utility = 0

    if plays > 3:
      terminalPass = (history[plays-1] == "p")
      doubleBet = (history[plays-2 : plays] == "bb")
      isPlayerCardHigher = (self.rank[history[player]] > self.rank[history[opponent]])

      if terminalPass:
        if history[-2:] == "pp":
          if isPlayerCardHigher:
            terminal_utility = 1
          else:
            terminal_utility = -1
        else:
            terminal_utility = 1
      elif doubleBet: #bb
          if isPlayerCardHigher:
            terminal_utility = 2
          else:
            terminal_utility = -2

    if player == target_player_i:
      return terminal_utility
    else:
      return -terminal_utility



  #terminal stateかどうかを判定
  def whether_terminal_states(self, history):
    """return string
    >>> KuhnTrainer().whether_terminal_states("")
    False
    >>> KuhnTrainer().whether_terminal_states("JK")
    False
    >>> KuhnTrainer().whether_terminal_states("JQpp")
    True
    >>> KuhnTrainer().whether_terminal_states("QKpb")
    False
    >>> KuhnTrainer().whether_terminal_states("JKbb")
    True
    >>> KuhnTrainer().whether_terminal_states("QKpbb")
    True
    """
    plays = len(history)
    if plays > 3:
      terminalPass = (history[plays-1] == "p")
      doubleBet = (history[plays-2 : plays] == "bb")

      return terminalPass or doubleBet
    else:
      return False


   #terminal stateかどうかを判定
  def whether_chance_node(self, history):
    """return string
    >>> KuhnTrainer().whether_chance_node("")
    True
    >>> KuhnTrainer().whether_chance_node("p")
    False
    """
    if history == "":
      return True
    else:
      return False


  # make node or get node
  def if_nonexistant(self, infoSet):
    if infoSet not in self.avg_strategy:
      self.avg_strategy[infoSet] = np.array([1/self.NUM_ACTIONS for _ in range(self.NUM_ACTIONS)], dtype=float)


  def calc_best_response_value(self, avg_strategy, best_response_strategy, best_response_player, history, prob):
      plays = len(history)
      player = plays % 2
      opponent = 1 - player

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        cards = np.array(["J", "Q", "K"])
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = cards_i[0] + cards_i[1]
          utility_sum +=  (1/len(cards_candicates))* self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob)
        return utility_sum

      infoSet = history[player] + history[2:]
      self.if_nonexistant(infoSet)


      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


          for assume_history, po_ in self.infoSets_dict[infoSet]:
            for ai in range(self.NUM_ACTIONS):
              nextHistory =  assume_history + ("p" if ai == 0 else "b")
              br_value[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, po_)
              action_value[ai] += br_value[ai] * po_

          br_action = 0
          for ai in range(self.NUM_ACTIONS):
            if action_value[ai] > action_value[br_action]:
              br_action = ai
          best_response_strategy[infoSet] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          best_response_strategy[infoSet][br_action] = 1.0


        node_util = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          node_util[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        for ai in range(self.NUM_ACTIONS):
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          action_value_list[ai] = self.calc_best_response_value(avg_strategy, best_response_strategy, best_response_player, nextHistory, prob*avg_strategy[infoSet][ai])
          nodeUtil += avg_strategy[infoSet][ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      cards = np.array(["J", "Q", "K"])
      cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
      for cards_i in cards_candicates:
        nextHistory = cards_i[0] + cards_i[1]
        self.create_infoSets(nextHistory, target_player, po)
      return

    infoSet = history[player] + history[2:]
    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = []
      self.infoSets_dict[infoSet].append((history, po))

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        self.if_nonexistant(infoSet)
        actionProb = self.avg_strategy[infoSet][ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)


  def get_exploitability_dfs(self):

    # 各information setを作成 & reach_probabilityを計算
    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        exploitability += self.calc_best_response_value(self.avg_strategy, best_response_strategy, best_response_player_i, "", 1)

    if exploitability < 0:
      return 1e-7
    #assert exploitability >= 0
    return exploitability


  def eval_vanilla_CFR(self, history, target_player_i, iteration_t, p0, p1):
      plays = len(history)
      player = plays % 2
      opponent = 1 - player

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, target_player_i)

      elif self.whether_chance_node(history):
        cards = np.array(["J", "Q", "K"])
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = cards_i[0] + cards_i[1]
          #regret　strategy 重み どのカードでも同じ確率
          utility_sum += (1/len(cards_candicates))* self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1)
        return utility_sum

      infoSet = history[player] + history[2:]
      self.if_nonexistant(infoSet)

      strategy = self.avg_strategy[infoSet]

      util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      nodeUtil = 0

      for ai in range(self.NUM_ACTIONS):
        nextHistory = history + ("p" if ai == 0 else "b")
        if player == 0:
          util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0 * strategy[ai], p1)
        else:
          util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1 * strategy[ai])
        nodeUtil += strategy[ai] * util_list[ai]

      return nodeUtil

  def show_plot(self, method):
    plt.scatter(list(self.exploitability_list.keys()), list(self.exploitability_list.values()), label=method)
    plt.plot(list(self.exploitability_list.keys()), list(self.exploitability_list.values()))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel("exploitability")
    plt.legend()


  #KuhnTrainer main method
  def train(self, n, m, memory_size):
    self.exploitability_list = {}

    self.M_SL_player0 = deque([], maxlen=memory_size)
    self.M_SL_player1 = deque([], maxlen=memory_size)
    self.M_RL_player0 = deque([], maxlen=memory_size)
    self.M_RL_player1 = deque([], maxlen=memory_size)

    self.M_SL = [self.M_SL_player0, self.M_SL_player1]
    self.M_RL = [self.M_RL_player0, self.M_RL_player1]


    # 各information setを作成 & reach_probabilityを計算 かつ self.ave_strategy を作成 (uniform strategy)
    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    self.best_response_strategy = copy.deepcopy(self.avg_strategy)

    # n_count
    self.N_count = copy.deepcopy(self.avg_strategy)
    for node, cn in self.N_count.items():
      self.N_count[node] = np.array([1.0 for _ in range(self.NUM_ACTIONS)], dtype=float)

    # q_value
    self.Q_value = [np.zeros((6,2)), np.zeros((6,2))]

    for iteration_t in tqdm(range(int(self.train_iterations))):
      eta = 1/(iteration_t+2)

      D = FSP_Kuhn_Poker_generate_data.GenerateData().generate_data(self.avg_strategy, self.best_response_strategy, n, m, eta)

      #self.best_response_strategy = {}
      #self.infoSets_dict = {}
      #for target_player in range(self.NUM_PLAYERS):
      #  self.create_infoSets("", target_player, 1.0)

      for player_i in range(self.NUM_ACTIONS):
        self.M_SL[player_i] = D[player_i]
        self.M_RL[player_i].extend(D[player_i])

        #self.calc_best_response_value(self.avg_strategy, self.best_response_strategy, player_i, "", 1)
        FSP_Kuhn_Poker_reinforcement_learning.ReinforcementLearning().RL_train(self.M_RL[player_i], player_i, self.best_response_strategy, self.Q_value[player_i], iteration_t)
        FSP_Kuhn_Poker_supervised_learning.SupervisedLearning().SL_train_AVG(self.M_SL[player_i], player_i, self.avg_strategy, self.N_count)
        #FSP_Kuhn_Poker_supervised_learning.SupervisedLearning().SL_train_MLP(self.M_SL[player_i], player_i, self.avg_strategy)


      if iteration_t in [int(j)-1 for j in np.logspace(0, len(str(self.train_iterations))-1, (len(str(self.train_iterations))-1)*3)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()
        wandb.log({'iteration': iteration_t, 'exploitability': self.exploitability_list[iteration_t]})

    self.show_plot("FSP")
    wandb.save()
