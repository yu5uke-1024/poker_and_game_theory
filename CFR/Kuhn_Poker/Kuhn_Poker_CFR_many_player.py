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
import math



#Node Class
class Node:
  #Kuhn_node_definitions
  def __init__(self, NUM_ACTIONS):
    self.NUM_ACTIONS = NUM_ACTIONS
    self.infoSet = None
    self.c = 0
    self.regretSum = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.strategy = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.strategySum = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.Get_strategy_through_regret_matching()


  #regret-matching
  def Get_strategy_through_regret_matching(self):
    self.normalizingSum = 0
    for a in range(self.NUM_ACTIONS):
      self.strategy[a] = self.regretSum[a] if self.regretSum[a]>0 else 0
      self.normalizingSum += self.strategy[a]

    for a in range(self.NUM_ACTIONS):
      if self.normalizingSum >0 :
        self.strategy[a] /= self.normalizingSum
      else:
        self.strategy[a] = 1/self.NUM_ACTIONS


  # calculate average-strategy
  def Get_average_information_set_mixed_strategy(self):
    self.avgStrategy = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.normalizingSum = 0
    for a in range(self.NUM_ACTIONS):
      self.normalizingSum += self.strategySum[a]

    for a in range(self.NUM_ACTIONS):
      if self.normalizingSum >0 :
        self.avgStrategy[a] = self.strategySum[a] / self.normalizingSum
      else:
        self.avgStrategy[a] = 1/ self.NUM_ACTIONS

    return self.avgStrategy



#Trainer class
class KuhnTrainer:
  def __init__(self, train_iterations=10**4, num_players =2):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.PASS = 0
    self.BET = 1
    self.NUM_ACTIONS = 2
    self.nodeMap = defaultdict(list)
    self.eval = False
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
      """
      pot = self.NUM_PLAYERS * 1 + history.count("b")
      start = -1
      target_player_action = history[self.NUM_PLAYERS+target_player_i::self.NUM_PLAYERS]

      # 全員が pass
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

      #target_plyaer は　pass , 誰かはbet
      elif ("b" not in target_player_action) and ("b" in history):
        return start

      else:
        #betしている +pot or -2
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


  #terminal stateかどうかを判定
  def whether_terminal_states(self, history):
    #passしかないhistory
    if "b" not in history:
      return history.count("p") == self.NUM_PLAYERS

    plays = len(history)
    first_bet = history.index("b")
    return plays - first_bet -1  == self.NUM_PLAYERS -1


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
  def Get_information_set_node_or_create_it_if_nonexistant(self, infoSet):
    node = self.nodeMap.get(infoSet)
    if node == None:
      node = Node(self.NUM_ACTIONS)
      self.nodeMap[infoSet] = node
    return node


  def Get_strategy(self, node):
    if not self.eval:
      strategy =  node.strategy
    else:
      if self.eval:
        strategy = node.Get_average_information_set_mixed_strategy()
    return strategy


  #chance sampling CFR
  def chance_sampling_CFR(self, history, target_player_i, iteration_t, p_list):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      nextHistory = "".join(cards[:self.NUM_PLAYERS])
      return self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p_list)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0
    node.Get_strategy_through_regret_matching()
    strategy = self.Get_strategy(node)

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)
      nodeUtil += strategy[ai] * util_list[ai]


    if (not self.eval) and  player == target_player_i:
      for ai in range(self.NUM_ACTIONS):
        regret = util_list[ai] - nodeUtil

        p_exclude = 1
        for idx in range(self.NUM_PLAYERS):
          if idx != player:
            p_exclude *= p_list[idx]

        node.regretSum[ai] += p_exclude * regret
        node.strategySum[ai] += strategy[ai] * p_list[player]

    return nodeUtil


  def vanilla_CFR(self, history, target_player_i, iteration_t, p_list):
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
        utility_sum +=  (1/len(cards_candicates))* self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)
      return utility_sum

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()

    strategy = self.Get_strategy(node)

    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)

      nodeUtil += strategy[ai] * util_list[ai]

    if (not self.eval) and  player == target_player_i:
      for ai in range(self.NUM_ACTIONS):
        regret = util_list[ai] - nodeUtil

        p_exclude = 1
        for idx in range(self.NUM_PLAYERS):
          if idx != player:
            p_exclude *= p_list[idx]

        node.regretSum[ai] += p_exclude * regret
        node.strategySum[ai] += strategy[ai] * p_list[player]

    return nodeUtil


  #external sampling MCCFR
  def external_sampling_MCCFR(self, history, target_player_i):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      nextHistory = "".join(cards[:self.NUM_PLAYERS])
      return self.external_sampling_MCCFR(nextHistory, target_player_i)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()

    if player == target_player_i:
      util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      nodeUtil = 0

      for ai in range(self.NUM_ACTIONS):
        nextHistory = history + ("p" if ai == 0 else "b")
        util_list[ai] = self.external_sampling_MCCFR(nextHistory, target_player_i)
        nodeUtil += node.strategy[ai] * util_list[ai]

      for ai in range(self.NUM_ACTIONS):
        regret = util_list[ai] - nodeUtil
        node.regretSum[ai] += regret

    else:
      sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p= node.strategy)
      nextHistory = history + ("p" if sampling_action == 0 else "b")
      nodeUtil= self.external_sampling_MCCFR(nextHistory, target_player_i)

      for ai in range(self.NUM_ACTIONS):
        node.strategySum[ai] += node.strategy[ai]

    return nodeUtil


  #outcome sampling MCCFR
  def outcome_sampling_MCCFR(self, history, target_player_i, iteration_t, p_list,s):
    plays = len(history)
    player = plays % self.NUM_PLAYERS

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i) / s, 1

    elif self.whether_chance_node(history):
      cards = self.card_distribution(self.NUM_PLAYERS)
      random.shuffle(cards)
      nextHistory = "".join(cards[:self.NUM_PLAYERS])
      return self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list, s)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()
    probability =  np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)

    if player == target_player_i:
      for ai in range(self.NUM_ACTIONS):
        probability[ai] =  self.epsilon/self.NUM_ACTIONS + (1-self.epsilon)* node.strategy[ai]
    else:
      for ai in range(self.NUM_ACTIONS):
        probability[ai] = node.strategy[ai]

    sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=probability)
    nextHistory = history + ("p" if sampling_action == 0 else "b")

    if player == target_player_i:

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = node.strategy[sampling_action]

      util, p_tail = self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list*p_change, s*probability[sampling_action])

      p_exclude = 1
      for idx in range(self.NUM_PLAYERS):
        if idx != player:
          p_exclude *= p_list[idx]

      w = util * p_exclude
      for ai in range(self.NUM_ACTIONS):
        if sampling_action == ai:
          regret = w*(1- node.strategy[sampling_action])*p_tail
        else:
          regret = -w*p_tail * node.strategy[sampling_action]
        node.regretSum[ai] +=  regret

    else:
        p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
        for idx in range(self.NUM_PLAYERS):
          if idx!= player:
            p_change[idx] = node.strategy[sampling_action]

        util, p_tail = self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list*p_change, s*probability[sampling_action])

        p_exclude = 1
        for idx in range(self.NUM_PLAYERS):
          if idx != player:
            p_exclude *= p_list[idx]

        for ai in range(self.NUM_ACTIONS):
          node.strategySum[ai] += (iteration_t - node.c)*p_exclude*node.strategy[ai]
        node.c = iteration_t
        #node.strategySum[ai] += (p1/s)*node.strategy[ai]

    return util, p_tail*node.strategy[sampling_action]


  #KuhnTrainer main method
  def train(self, method):
    self.exploitability_list = {}
    for iteration_t in tqdm(range(int(self.train_iterations))):
      for target_player_i in range(self.NUM_PLAYERS):

        p_list = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)

        if method == "vanilla_CFR":
          self.vanilla_CFR("", target_player_i, iteration_t, p_list)
        elif method == "chance_sampling_CFR":
          self.chance_sampling_CFR("", target_player_i, iteration_t, p_list)
        elif method == "external_sampling_MCCFR":
          self.external_sampling_MCCFR("", target_player_i)
        elif method == "outcome_sampling_MCCFR":
          self.epsilon = 0.6
          self.outcome_sampling_MCCFR("", target_player_i, iteration_t, p_list, 1)

      #calculate expolitability
      if iteration_t in [int(j)-1 for j in np.logspace(1, len(str(self.train_iterations))-1, (len(str(self.train_iterations))-1)*3)] :
        self.exploitability_list[iteration_t] = self.get_exploitability_dfs()

    #self.show_plot(method)


  def show_plot(self, method):
    plt.scatter(list(self.exploitability_list.keys()), list(self.exploitability_list.values()), label=method)
    plt.plot(list(self.exploitability_list.keys()), list(self.exploitability_list.values()))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel("exploitability")
    plt.legend(loc = "lower left")
    plt.show()


  # evaluate average strategy
  def eval_strategy(self, target_player_i):
    self.eval = True
    p_list = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
    average_utility = self.vanilla_CFR("", target_player_i, 0, p_list)
    self.eval = False
    return average_utility


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
      node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)

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
        avg_strategy = node.Get_average_information_set_mixed_strategy()
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in range(self.NUM_ACTIONS):
          nextHistory =  history + ("p" if ai == 0 else "b")
          action_value_list[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob*avg_strategy[ai])
          nodeUtil += avg_strategy[ai] * action_value_list[ai]
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
      self.infoSets_dict[infoSet].append((history, po))

    for ai in range(self.NUM_ACTIONS):
      nextHistory = history + ("p" if ai == 0 else "b")
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
        actionProb = node.Get_average_information_set_mixed_strategy()[ai]
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

    if exploitability < 0:
      return 1e-7
    #assert exploitability >= 0
    return exploitability

"""
kuhn_trainer = KuhnTrainer(train_iterations=10**4, num_players=2)
kuhn_trainer.train("vanilla_CFR")
#kuhn_trainer.train("chance_sampling_CFR")
#kuhn_trainer.train("external_sampling_MCCFR")
#kuhn_trainer.train("outcome_sampling_MCCFR")

print("avg util:", kuhn_trainer.eval_strategy(0))

result_dict = {}
for key, value in sorted(kuhn_trainer.nodeMap.items()):
  result_dict[key] = value.Get_average_information_set_mixed_strategy()
df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=['Pass', "Bet"])
df.index.name = "Node"
print(df)
"""

# 特定のstrategy_profileのexploitabilityを計算する
# all fold strategy_profile → 2
#kuhn_poker_agent = KuhnTrainer(train_iterations=0, num_players=2)
#kuhn_poker_agent.eval_strategy(0)
#for node_i, node_main in kuhn_poker_agent.nodeMap.items():
#  kuhn_poker_agent.nodeMap[node_i].strategySum =  np.array([1, 0], dtype=float)
#print(kuhn_poker_agent.get_exploitability_dfs())

print("")
# random strategy_profileのexploitability
#→0.9166666666666665
for i in range(2,6):
  kuhn_poker_agent = KuhnTrainer(train_iterations=0, num_players=i)
  print("{}人対戦:".format(i), kuhn_poker_agent.get_exploitability_dfs())


doctest.testmod()
