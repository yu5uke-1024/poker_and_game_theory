#Library
from typing import ByteString
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


import FSP_Kuhn_Poker_supervised_learning
import FSP_Kuhn_Poker_reinforcement_learning
import FSP_Kuhn_Poker_generate_data



class KuhnTrainer:
  def __init__(self, train_iterations=10**1, num_players =2):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.NUM_ACTIONS = 2
    self.avg_strategy = {}
    self.card_rank = self.make_rank(self.NUM_PLAYERS)


  def make_rank(self, num_players):
    """return dict
    >>> KuhnTrainer().make_rank(2) == {'J':1, 'Q':2, 'K':3}
    True
    """
    card_rank = {}
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    for i in range(num_players+1):
      card_rank[card[11-num_players+i]] =  i+1
    return card_rank


  def card_distribution(self, num_players):
    """return list
    >>> KuhnTrainer().card_distribution(2)
    ['J', 'Q', 'K']
    """
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    return card[11-num_players:]


  #return util for terminal state for target_player_i
  def Return_payoff_for_terminal_states(self, history, target_player_i):
      """return list
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKbb", 0)
      -2
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKbb", 1)
      2
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpp", 0)
      -1
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpp", 1)
      1
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpbp", 0)
      -1
      >>> KuhnTrainer(num_players=2).Return_payoff_for_terminal_states("JKpbp", 1)
      1
      >>> KuhnTrainer(num_players=3).Return_payoff_for_terminal_states("JKTpbpp", 1)
      2
      """

      pot = self.NUM_PLAYERS * 1 + history.count("b")
      start = -1
      target_player_action = history[self.NUM_PLAYERS+target_player_i::self.NUM_PLAYERS]

      # all players pass
      if ("b" not in history) and (history.count("p") == self.NUM_PLAYERS):
        pass_player_card = {}
        for idx in range(self.NUM_PLAYERS):
          pass_player_card[idx] = [history[idx], self.card_rank[history[idx]]]

        winner_rank = max([idx[1] for idx in pass_player_card.values()])

        target_player_rank = pass_player_card[target_player_i][1]

        if target_player_rank == winner_rank:
          return start + pot
        else:
          return start

      #target plyaer do pass , another player do bet
      elif ("b" not in target_player_action) and ("b" in history):
        return start

      else:
        #bet → +pot or -2
        bet_player_list = [idx%self.NUM_PLAYERS for idx, act in enumerate(history[self.NUM_PLAYERS:]) if act == "b"]
        bet_player_card = {}
        for idx in bet_player_list:
          bet_player_card[idx] = [history[idx], self.card_rank[history[idx]]]


        winner_rank = max([idx[1] for idx in bet_player_card.values()])
        target_player_rank = bet_player_card[target_player_i][1]
        if target_player_rank == winner_rank:
          return start + pot - 1
        else:
          return start - 1


  #whether terminal state
  def whether_terminal_states(self, history):
    #pass only history
    if "b" not in history:
      return history.count("p") == self.NUM_PLAYERS

    plays = len(history)
    first_bet = history.index("b")
    return plays - first_bet -1  == self.NUM_PLAYERS -1


  #whether chance node state
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


  def calc_best_response_value(self, best_response_strategy, best_response_player, history, prob):
      plays = len(history)
      player = plays % self.NUM_PLAYERS

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        cards = self.card_distribution(self.NUM_PLAYERS)
        cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
          utility_sum +=  (1/len(cards_candicates))* self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        return utility_sum

      infoSet = history[player] + history[self.NUM_PLAYERS:]
      self.if_nonexistant(infoSet)

      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


          for assume_history, po_ in self.infoSets_dict[infoSet]:
            for ai in range(self.NUM_ACTIONS):
              nextHistory =  assume_history + ("p" if ai == 0 else "b")
              br_value[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, po_)
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
          node_util[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        for ai in range(self.NUM_ACTIONS):
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          action_value_list[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob* self.avg_strategy[infoSet][ai])
          nodeUtil += self.avg_strategy[infoSet][ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
        self.create_infoSets(nextHistory, target_player, po)
      return

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = []
        self.infoSets_dict_player[player].append(infoSet)
      self.infoSets_dict[infoSet].append((history, po))


    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        self.if_nonexistant(infoSet)
        actionProb =self.avg_strategy[infoSet][ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)




  def get_exploitability_dfs(self):

    # 各information setを作成 & reach_probabilityを計算
    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        exploitability += self.calc_best_response_value(best_response_strategy, best_response_player_i, "", 1)

    assert exploitability >= 0
    return exploitability



  def eval_vanilla_CFR(self, history, target_player_i, iteration_t, p_list):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      cards_candicates = [list(cards_candicate) for cards_candicate in itertools.permutations(cards)]
      utility_sum = 0
      for cards_i in cards_candicates:
        nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
        utility_sum +=  (1/len(cards_candicates))* self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)
      return utility_sum

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    self.if_nonexistant(infoSet)

    strategy = self.avg_strategy[infoSet]

    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.eval_vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)

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
    #plt.show()



  #KuhnTrainer main method
  def train(self, n, m, memory_size_rl, memory_size_sl, wandb_save, rl_algo, sl_algo, pseudo_code):
    self.exploitability_list = {}

    self.M_SL = [deque([], maxlen=memory_size_sl) for _ in range(self.NUM_PLAYERS)]
    self.M_RL = [deque([], maxlen=memory_size_rl) for _ in range(self.NUM_PLAYERS)]

    self.infoSets_dict_player = [[] for _ in range(self.NUM_PLAYERS)]
    self.infoSets_dict = {}

    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    self.best_response_strategy = copy.deepcopy(self.avg_strategy)

    # n_count
    self.N_count = copy.deepcopy(self.avg_strategy)
    for node, cn in self.N_count.items():
      self.N_count[node] = np.array([1.0 for _ in range(self.NUM_ACTIONS)], dtype=float)

    # q_value
    self.Q_value = [np.zeros((len(self.infoSets_dict_player[i]),2)) for i in range(self.NUM_PLAYERS)]



    RL = FSP_Kuhn_Poker_reinforcement_learning.ReinforcementLearning(self.infoSets_dict_player, self.NUM_PLAYERS, self.NUM_ACTIONS)
    SL = FSP_Kuhn_Poker_supervised_learning.SupervisedLearning(self.NUM_PLAYERS, self.NUM_ACTIONS)
    GD = FSP_Kuhn_Poker_generate_data.GenerateData(self.NUM_PLAYERS, self.NUM_ACTIONS)


    for iteration_t in tqdm(range(int(self.train_iterations))):

      if pseudo_code == "batch_FSP":
        GD.generate_data1(self.avg_strategy, n, self.M_RL)

        for player_i in range(self.NUM_PLAYERS):
          RL.RL_train(self.M_RL[player_i], player_i, self.best_response_strategy, self.Q_value[player_i], iteration_t, rl_algo)

        GD.generate_data2(self.avg_strategy, self.best_response_strategy, m, self.M_RL, self.M_SL)

        for player_i in range(self.NUM_PLAYERS):
          if sl_algo == "cnt":
            SL.SL_train_AVG(self.M_SL[player_i], player_i, self.avg_strategy, self.N_count)
            self.M_SL[player_i] = []
          elif sl_algo == "mlp":
            SL.SL_train_MLP(self.M_SL[player_i], player_i, self.avg_strategy)


      elif pseudo_code == "general_FSP":
        eta = 1/(iteration_t+2)
        D = GD.generate_data0(self.avg_strategy, self.best_response_strategy, n, m, eta)
        for player_i in range(self.NUM_PLAYERS):
          self.M_SL[player_i].extend(D[player_i])
          self.M_RL[player_i].extend(D[player_i])

          RL.RL_train(self.M_RL[player_i], player_i, self.best_response_strategy, self.Q_value[player_i], iteration_t, rl_algo)

          if sl_algo == "cnt":
            SL.SL_train_AVG(self.M_SL[player_i], player_i, self.avg_strategy, self.N_count)
          elif sl_algo == "mlp":
            SL.SL_train_MLP(self.M_SL[player_i], player_i, self.avg_strategy)


      if iteration_t in [int(j)-1 for j in np.logspace(0, len(str(self.train_iterations))-1, (len(str(self.train_iterations))-1)*3)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()

        if wandb_save:
          wandb.log({'iteration': iteration_t, 'exploitability': self.exploitability_list[iteration_t]})

    self.show_plot("FSP")
    if wandb_save:
      wandb.save()


doctest.testmod()
