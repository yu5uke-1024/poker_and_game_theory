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


#Node Class
#information set node class definition
class Node:
  #1 #Leduc_node_definitions
  def __init__(self, NUM_ACTIONS, infoSet):
    self.NUM_ACTIONS = NUM_ACTIONS
    self.infoSet = infoSet
    self.c = 0
    self.regretSum = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.strategy = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.strategySum = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.possible_action = self.Get_possible_action_by_information_set()
    self.Get_strategy_through_regret_matching()


  def Get_possible_action_by_information_set(self): #{0:"f", 1:"c", 2:"r"}
    """return int
    >>> Node(3, "JccKc").Get_possible_action_by_information_set()
    array([1, 2])
    >>> Node(3, "Jr").Get_possible_action_by_information_set()
    array([0, 1, 2])
    >>> Node(3, "JccJc").Get_possible_action_by_information_set()
    array([1, 2])
    >>> Node(3, "J").Get_possible_action_by_information_set()
    array([1, 2])
    """
    infoset_without_hand_card = self.infoSet[1:]
    if ("J" in infoset_without_hand_card) or ("Q" in infoset_without_hand_card) or ("K" in infoset_without_hand_card):
      private_cards, history_before, community_card, history_after = LeducTrainer().Split_history("??" + infoset_without_hand_card)
      infoset_without_hand_card = history_after

    if infoset_without_hand_card == "" or infoset_without_hand_card == "c":
      return np.array([1,2], dtype=int)
    elif infoset_without_hand_card == "cr" or infoset_without_hand_card == "r":
      return np.array([0,1,2], dtype=int)
    elif infoset_without_hand_card == "crr" or infoset_without_hand_card == "rr":
      return np.array([0,1], dtype=int)

  #regret-matching
  def Get_strategy_through_regret_matching(self):
    self.normalizingSum = 0
    for a in self.possible_action:
      self.strategy[a] = self.regretSum[a] if self.regretSum[a]>0 else 0
      self.normalizingSum += self.strategy[a]

    for a in self.possible_action:
      if self.normalizingSum >0 :
        self.strategy[a] /= self.normalizingSum
      else:
        self.strategy[a] = 1/len(self.possible_action)

  # calculate average-strategy
  def Get_average_information_set_mixed_strategy(self):
    self.avgStrategy = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    self.normalizingSum = 0
    for a in self.possible_action:
      self.normalizingSum += self.strategySum[a]

    for a in self.possible_action:
      if self.normalizingSum >0 :
        self.avgStrategy[a] = self.strategySum[a] / self.normalizingSum
      else:
        self.avgStrategy[a] = 1/len(self.possible_action)

    return self.avgStrategy

# Leduc Trainer
class LeducTrainer:
  def __init__(self, iterations=10000):
    #f: FOLD, c: CALL , r:RAISE
    self.iterations = iterations
    self.NUM_PLAYERS = 2
    self.ACTION_DICT = {0:"f", 1:"c", 2:"r"}
    self.NUM_ACTIONS = 3
    self.nodeMap = defaultdict(list)
    self.eval = None


  def Rank(self, my_card, com_card):
    """return int
    >>> LeducTrainer().Rank("J", "Q")
    1
    >>> LeducTrainer().Rank("K", "K")
    6
    """
    rank = {"KK":6, "QQ":5, "JJ":4, "KQ":3, "QK":3, "KJ":2, "JK":2, "QJ":1, "JQ":1}
    hand = my_card + com_card
    return rank[hand]


  def Split_history(self, history):
    """return history_before, history_after
    >>> LeducTrainer().Split_history("JKccKcrc")
    ('JK', 'cc', 'K', 'crc')
    >>> LeducTrainer().Split_history("KQrrcQrrc")
    ('KQ', 'rrc', 'Q', 'rrc')
    >>> LeducTrainer().Split_history("QQcrrcKcc")
    ('QQ', 'crrc', 'K', 'cc')
    """
    for ai, history_ai in enumerate(history[2:]):
      if history_ai == "J" or history_ai == "Q" or history_ai == "K":
        idx = ai+2
        community_catd = history_ai
    return history[:2], history[2:idx], community_catd, history[idx+1:]


  def Calculate_pot_size(self, round1_history):
    """return int
    >>> LeducTrainer().Calculate_pot_size("cc")
    1
    >>> LeducTrainer().Calculate_pot_size("rrc")
    4
    """
    if round1_history == "cc":  return +1
    elif round1_history == "crc" or round1_history == "rc": return +2
    elif round1_history == "crrc" or round1_history == "rrc": return +4


  #6 Return payoff for terminal states #if terminal states  return util
  def Return_payoff_for_terminal_states(self, history, target_player_i):
    """return int
    >>> LeducTrainer().Return_payoff_for_terminal_states("JJccQcc", 0)
    0
    >>> LeducTrainer().Return_payoff_for_terminal_states("JQcrcKcrc", 0)
    -6
    >>> LeducTrainer().Return_payoff_for_terminal_states("JQcrcKcrc", 1)
    6
    >>> LeducTrainer().Return_payoff_for_terminal_states("KQrf", 0)
    1
    >>> LeducTrainer().Return_payoff_for_terminal_states("QKcrf", 0)
    -1
    >>> LeducTrainer().Return_payoff_for_terminal_states("QKrrcQrrf", 0)
    -8
    >>> LeducTrainer().Return_payoff_for_terminal_states("QKrrcQrrc", 0)
    12
    """

    if len(history) >= 2 and ("J" in history[2:]) or ("Q" in history[2:]) or ("K" in history[2:]):
      private_cards, history_before, community_card, history_after = self.Split_history(history)
      self.round1_pot = self.Calculate_pot_size(history_before)
      plays = len(history_after)
      player = plays % 2
      opponent = 1 - player
      total_pot = 0
      if history_after[-1] == "f":
        if len(history_after) == 2: total_pot = self.round1_pot
        elif len(history_after) == 4: total_pot = +4 + self.round1_pot
        elif len(history_after) == 3:
          if history_after == "crf": total_pot = self.round1_pot
          elif history_after == "rrf": total_pot = +4 + self.round1_pot

      elif history_after[-2:] == "cc" or history_after[-2:] == "rc":
        if self.Rank(history[player], community_card) == self.Rank(history[opponent], community_card):
          return 0

        isPlayerCardStronger = (self.Rank(history[player], community_card) > self.Rank(history[opponent], community_card))

        if history_after == "cc":
          if isPlayerCardStronger: total_pot = self.round1_pot
          else: total_pot = -self.round1_pot
        elif history_after == "crc":
          if isPlayerCardStronger: total_pot = +4+self.round1_pot
          else: total_pot = -4-self.round1_pot
        elif history_after == "crrc":
          if isPlayerCardStronger: total_pot = +8+self.round1_pot
          else: total_pot = -8-self.round1_pot
        elif history_after == "rc":
          if isPlayerCardStronger: total_pot = +4+self.round1_pot
          else: total_pot = -4-self.round1_pot
        elif history_after == "rrc":
          if isPlayerCardStronger: total_pot = +8+self.round1_pot
          else: total_pot = -8-self.round1_pot

    elif history[-1] == "f":
      plays = len(history)
      player = plays % 2
      opponent = 1 - player
      if len(history) == 4: total_pot = +1
      elif len(history) == 6: total_pot = +2
      elif len(history) == 5:
        if history[2:] == "crf": total_pot = +1
        elif history[2:] == "rrf": total_pot = +2

    if player == target_player_i:
      return total_pot
    else:
      return -total_pot


  # whetther terminal_states
  def whether_terminal_states(self, history):
    """return string
    >>> LeducTrainer().whether_terminal_states("JKccKr")
    False
    >>> LeducTrainer().whether_terminal_states("QJccJcc")
    True
    >>> LeducTrainer().whether_terminal_states("QQcr")
    False
    >>> LeducTrainer().whether_terminal_states("QKrf")
    True
    >>> LeducTrainer().whether_terminal_states("KKccQcrf")
    True
    """
    if len(history) >= 2 and ("J" in history[2:]) or ("Q" in history[2:]) or ("K" in history[2:]):
      if history[-1] == "f":
        return True
      elif history[-2:] == "cc" or history[-2:] == "rc":
        return True
      else:
        return False
    else:
      if len(history) >=3 and history[-1] == "f":
        return True
      else:
        return False


  def whether_chance_node(self, history):
    """return string
    >>> LeducTrainer().whether_chance_node("JKcc")
    True
    >>> LeducTrainer().whether_chance_node("KQcr")
    False
    >>> LeducTrainer().whether_chance_node("")
    True
    >>> LeducTrainer().whether_chance_node("p")
    False
    """
    if history == "":
      return True
    elif history[2:] == "cc" or  history[2:] == "crc" or  history[2:] == "crrc" or  history[2:] == "rc" or  history[2:] == "rrc":
      return True
    else:
      return False


  #make node or get node
  def Get_information_set_node_or_create_it_if_nonexistant(self, infoSet):
    node = self.nodeMap.get(infoSet)
    if node == None:
      node = Node(self.NUM_ACTIONS, infoSet)
      self.nodeMap[infoSet] = node
    return node


  #chance sampling CFR
  def chance_sampling_CFR(self, history, target_player_i, iteration_t, p0, p1):
    #print(history, target_player_i, iteration_t, p0, p1)
    plays = len(history)
    if len(history) >= 2 and ("J" in history[2:]) or ("Q" in history[2:]) or ("K" in history[2:]):
      private_cards, history_before, community_card, history_after = self.Split_history(history)
      plays = len(history_after)

    player = plays % 2
    opponent = 1 - player

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      if len(history) == 0:
        self.cards = np.array(["J","J", "Q","Q", "K", "K"])
        random.shuffle(self.cards)
        nextHistory = self.cards[0] + self.cards[1]
        return self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p0, p1)
      else:
        nextHistory = history + self.cards[2]
        return self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p0, p1)

    infoSet = history[player] + history[2:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0


    for ai in node.possible_action:
      nextHistory = history + self.ACTION_DICT[ai]
      if player == 0:
        util_list[ai] = self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p0 * node.strategy[ai], p1)
      else:
        util_list[ai] = self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p0, p1 * node.strategy[ai])
      nodeUtil += node.strategy[ai] * util_list[ai]

    if player == target_player_i:
      for ai in node.possible_action:
        regret = util_list[ai] - nodeUtil
        node.regretSum[ai] += (p1 if player==0 else p0) * regret
        node.strategySum[ai] += node.strategy[ai] * (p0 if player==0 else p1)
      node.Get_strategy_through_regret_matching()

    return nodeUtil


  #chance sampling CFR
  def vanilla_CFR(self, history, target_player_i, iteration_t, p0, p1):
    plays = len(history)
    if len(history) >= 2 and ("J" in history[2:]) or ("Q" in history[2:]) or ("K" in history[2:]):
      private_cards, history_before, community_card, history_after = self.Split_history(history)
      plays = len(history_after)

    player = plays % 2
    opponent = 1 - player

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      if len(history) == 0:
        cards = np.array(["J","J", "Q","Q", "K", "K"])
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards)]
        utility_sum = 0
        for cards_i in cards_candicates:
          self.cards_i = cards_i
          nextHistory = self.cards_i[0] + self.cards_i[1]
          utility_sum += (1/len(cards_candicates))* self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1)
        return  utility_sum
      else:
        nextHistory = history + self.cards_i[2]
        return self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1)

    infoSet = history[player] + history[2:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0

    if not self.eval:
      strategy =  node.strategy
    else:
      strategy = node.Get_average_information_set_mixed_strategy()

    for ai in node.possible_action:
      nextHistory = history + self.ACTION_DICT[ai]
      if player == 0:
        util_list[ai] = self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p0 * strategy[ai], p1)
      else:
        util_list[ai] = self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p0, p1 * strategy[ai])
      nodeUtil += strategy[ai] * util_list[ai]

    if (not self.eval) and  player == target_player_i:
      for ai in node.possible_action:
        regret = util_list[ai] - nodeUtil
        node.regretSum[ai] += (p1 if player==0 else p0) * regret
        node.strategySum[ai] += strategy[ai] * (p0 if player==0 else p1)
      node.Get_strategy_through_regret_matching()

    return nodeUtil



  #external sampling MCCFR
  def external_sampling_MCCFR(self, history, target_player_i):
    plays = len(history)
    if len(history) >= 2 and ("J" in history[2:]) or ("Q" in history[2:]) or ("K" in history[2:]):
      private_cards, history_before, community_card, history_after = self.Split_history(history)
      plays = len(history_after)

    player = plays % 2
    opponent = 1 - player

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
          if len(history) == 0:
            self.cards = np.array(["J","J", "Q","Q", "K", "K"])
            random.shuffle(self.cards)
            nextHistory = self.cards[0] + self.cards[1]
            return self.external_sampling_MCCFR(nextHistory, target_player_i)
          else:
            nextHistory = history + self.cards[2]
            return self.external_sampling_MCCFR(nextHistory, target_player_i)


    infoSet = history[player] + history[2:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()

    if player == target_player_i:
      util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
      nodeUtil = 0

      for ai in node.possible_action:
        nextHistory = history + self.ACTION_DICT[ai]
        util_list[ai] = self.external_sampling_MCCFR(nextHistory, target_player_i)
        nodeUtil += node.strategy[ai] * util_list[ai]

      for ai in node.possible_action:
        regret = util_list[ai] - nodeUtil
        node.regretSum[ai] += regret

    else:
      sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p= node.strategy)
      nextHistory = history + self.ACTION_DICT[sampling_action]
      nodeUtil= self.external_sampling_MCCFR(nextHistory, target_player_i)

      for ai in node.possible_action:
        node.strategySum[ai] += node.strategy[ai]

    return nodeUtil



  #outcome sampling MCCFR
  def outcome_sampling_MCCFR(self, history, target_player_i, iteration_t, p0, p1,s):
    plays = len(history)
    if len(history) >= 2 and ("J" in history[2:]) or ("Q" in history[2:]) or ("K" in history[2:]):
      private_cards, history_before, community_card, history_after = self.Split_history(history)
      plays = len(history_after)

    player = plays % 2
    opponent = 1 - player

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i) / s, 1

    elif self.whether_chance_node(history):
          if len(history) == 0:
            self.cards = np.array(["J","J", "Q","Q", "K", "K"])
            random.shuffle(self.cards)
            nextHistory = self.cards[0] + self.cards[1]
            return self.outcome_sampling_MCCFR(nextHistory, target_player_i)
          else:
            nextHistory = history + self.cards[2]
            return self.outcome_sampling_MCCFR(nextHistory, target_player_i)

    infoSet = history[player] + history[2:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()
    probability =  np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)

    if player == target_player_i:
      for ai in node.possible_action:
        probability[ai] =  self.epsilon/self.NUM_ACTIONS + (1-self.epsilon)* node.strategy[ai]
    else:
      for ai in node.possible_action:
        probability[ai] = node.strategy[ai]


    sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=probability)
    nextHistory = history + self.ACTION_DICT[sampling_action]


    if player == target_player_i:
      util, p_tail = self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p0*node.strategy[sampling_action], p1, s*probability[sampling_action])

      w = util * p1
      for ai in node.possible_action:
        if sampling_action == ai:
          regret = w*(1- node.strategy[sampling_action])*p_tail
        else:
          regret = -w*p_tail * node.strategy[sampling_action]
        node.regretSum[ai] +=  regret


    else:
        util, p_tail = self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p0, p1*node.strategy[sampling_action], s*probability[sampling_action])

        for ai in node.possible_action:
          node.strategySum[ai] += (iteration_t - node.c)*p1*node.strategy[ai]
        node.c = iteration_t
        #node.strategySum[ai] += (p1/s)*node.strategy[ai]

    return util, p_tail*node.strategy[sampling_action]

  #KuhnTrainer main method
  def train(self, method):
    self.eval = False
    for iteration_t in tqdm(range(int(self.iterations))):
      for target_player_i in range(self.NUM_PLAYERS):
        self.chance_sampling_CFR("", target_player_i, iteration_t, 1, 1)
        #pass



  # evaluate average strategy
  def eval_strategy(self):
    self.eval = True
    target_player_i = 0
    average_utility = 0
    average_utility = self.vanilla_CFR("", target_player_i, 0, 1, 1)

    print("")
    print("average eval util:", average_utility)




#学習
leduc_trainer = LeducTrainer(iterations=100000)
#leduc_trainer.train("vanilla_CFR")
leduc_trainer.train("chance_sampling_CFR")
#leduc_trainer.train("external_sampling_MCCFR")
#leduc_trainer.train("outcome_sampling_MCCFR")


print("avg util:", leduc_trainer.eval_strategy())



pd.set_option('display.max_rows', None)
result_dict = {}
for key, value in sorted(leduc_trainer.nodeMap.items()):
  result_dict[key] = value.Get_average_information_set_mixed_strategy()

df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=["Fold", "Call", "Raise"])
df.index.name = "Node"
df

print(df)



doctest.testmod()
