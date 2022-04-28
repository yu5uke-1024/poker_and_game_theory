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
  def __init__(self, NUM_ACTIONS, infoSet, num_players=2):
    self.NUM_ACTIONS = NUM_ACTIONS
    self.NUM_PLAYERS = num_players
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
    if LeducTrainer().card_num_check(infoset_without_hand_card) == 1:
      private_cards, history_before, community_card, history_after = LeducTrainer().Split_history("??" + infoset_without_hand_card)
      infoset_without_hand_card = history_after

    if  len(infoset_without_hand_card) == 0 or infoset_without_hand_card.count("r") == 0:
      return np.array([1,2], dtype=int)
    elif infoset_without_hand_card.count("r") == 1:
      return np.array([0,1,2], dtype=int)
    elif infoset_without_hand_card.count("r") == 2:
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
  def __init__(self, train_iterations=10, num_players = 2):
    #f: FOLD, c: CALL , r:RAISE
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.ACTION_DICT = {0:"f", 1:"c", 2:"r"}
    self.NUM_ACTIONS = 3
    self.nodeMap = defaultdict(list)
    self.eval = None
    self.card_rank = self.make_rank()

  def make_rank(self):
    """return dict
    >>> LeducTrainer(num_players=2).make_rank() == {"KK":6, "QQ":5, "JJ":4, "KQ":3, "QK":3, "KJ":2, "JK":2, "QJ":1, "JQ":1}
    True
    """
    card_deck = self.card_distribution()
    card_unique = card_deck[::2]
    card_rank = {}
    count = (len(card_unique)-1)*len(card_unique) //2
    for i in range(len(card_unique)-1,-1, -1):
      for j in range(i-1, -1, -1):
            card_rank[card_unique[i] + card_unique[j]] = count
            card_rank[card_unique[j] + card_unique[i]] = count
            count -= 1

    count = (len(card_unique)-1)*len(card_unique) //2 +1
    for i in range(len(card_unique)):
        card_rank[card_unique[i] + card_unique[i]] = count
        count += 1

    return card_rank


  def Rank(self, my_card, com_card):
    """return int
    >>> LeducTrainer(num_players=2).Rank("J", "Q")
    1
    >>> LeducTrainer(num_players=2).Rank("K", "K")
    6
    """
    hand = my_card + com_card
    return self.card_rank[hand]


  def card_distribution(self):
    """return list
    >>> LeducTrainer(num_players=2).card_distribution()
    ['J', 'J', 'Q', 'Q', 'K', 'K']
    >>> LeducTrainer(num_players=3).card_distribution()
    ['T', 'T', 'J', 'J', 'Q', 'Q', 'K', 'K']
    """
    card = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    card_deck = []
    for i in range(self.NUM_PLAYERS+1):
      card_deck.append(card[11-self.NUM_PLAYERS+i])
      card_deck.append(card[11-self.NUM_PLAYERS+i])

    return card_deck


  # round2に入っているhistoryを 手元card, round1 action, community card, round2 cardに分ける
  def Split_history(self, history):
    """return history_before, history_after
    >>> LeducTrainer(num_players=3).Split_history("JKQcccKcrcc")
    ('JKQ', 'ccc', 'K', 'crcc')
    >>> LeducTrainer(num_players=2).Split_history("KQrrcQrrc")
    ('KQ', 'rrc', 'Q', 'rrc')
    >>> LeducTrainer(num_players=2).Split_history("QQcrrcKcc")
    ('QQ', 'crrc', 'K', 'cc')
    """
    for ai, history_ai in enumerate(history[self.NUM_PLAYERS:]):
      if history_ai in self.card_distribution():
        idx = ai+self.NUM_PLAYERS
        community_catd = history_ai
    return history[:self.NUM_PLAYERS], history[self.NUM_PLAYERS:idx], community_catd, history[idx+1:]


  def action_history_player(self, history):
    #target_player_iのaction 履歴
    player_action_list = [[] for _ in range(self.NUM_PLAYERS)]
    player_money_list_round1 = [1 for _ in range(self.NUM_PLAYERS)]
    player_money_list_round2 = [0 for _ in range(self.NUM_PLAYERS)]

    f_count, a_count, raise_count = 0, 0, 0

    card = self.card_distribution()
    private_cards, history_before, community_card, history_after = self.Split_history(history)
    for hi in history_before:
      while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
        f_count += 1
      player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)

      if hi == "c":
        player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] = max(player_money_list_round1)
      elif hi == "r" and raise_count == 0:
        raise_count += 1
        player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 1
      elif hi == "r" and raise_count == 1:
        player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 3

      a_count += 1

    f_count, a_count, raise_count = 0, 0, 0

    for hi in history_after:
        if hi not in card:
          while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
            f_count += 1
          player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)

          if hi == "c":
            player_money_list_round2[(a_count + f_count)%self.NUM_PLAYERS] = max(player_money_list_round2)
          elif hi == "r" and raise_count == 0:
            raise_count += 1
            player_money_list_round2[(a_count + f_count)%self.NUM_PLAYERS] += 4
          elif hi == "r" and raise_count == 1:
            player_money_list_round2[(a_count + f_count)%self.NUM_PLAYERS] += 8

          a_count += 1

    return player_action_list, player_money_list_round1, player_money_list_round2, community_card

  def action_player(self, history):
    """return int
    >>> LeducTrainer().action_player("JJc")
    1
    >>> LeducTrainer().action_player("JQcr")
    0
    >>> LeducTrainer(num_players=3).action_player("JQTrfr")
    0
    """
    player_action_list = [[] for _ in range(self.NUM_PLAYERS)]
    a_count = 0
    f_count = 0

    if self.card_num_check(history) == self.NUM_PLAYERS:

      for hi in history[self.NUM_PLAYERS:]:
        while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
          f_count += 1
        player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)
        a_count += 1
    elif self.card_num_check(history) == self.NUM_PLAYERS+1:

      private_cards, history_before, community_card, history_after = self.Split_history(history)
      for hi in history_after:
        while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
          f_count += 1
        player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)
        a_count += 1

    return (a_count + f_count)%self.NUM_PLAYERS


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
    #round1で終了
    if history.count("f") == self.NUM_PLAYERS -1  and self.card_num_check(history) == self.NUM_PLAYERS:
      player_action_list = [[] for _ in range(self.NUM_PLAYERS)]
      player_money_list_round1 = [1 for _ in range(self.NUM_PLAYERS)]
      player_money_list_round2 = [0 for _ in range(self.NUM_PLAYERS)]

      f_count, a_count, raise_count = 0, 0, 0

      for hi in history[self.NUM_PLAYERS:]:
        while len(player_action_list[(a_count + f_count)%self.NUM_PLAYERS])>=1 and player_action_list[(a_count + f_count)%self.NUM_PLAYERS][-1] == "f":
          f_count += 1
        player_action_list[(a_count + f_count)%self.NUM_PLAYERS].append(hi)

        if hi == "c":
          player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] = max(player_money_list_round1)
        elif hi == "r" and raise_count == 0:
          raise_count += 1
          player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 1
        elif hi == "r" and raise_count == 1:
          player_money_list_round1[(a_count + f_count)%self.NUM_PLAYERS] += 3

        a_count += 1
      if len(player_action_list[target_player_i]) >= 1 and player_action_list[target_player_i][-1] == "f":
        return -player_money_list_round1[target_player_i]
      else:
        return sum(player_money_list_round1) -player_money_list_round1[target_player_i]

    #round2で終了
    #target_player_iのaction 履歴
    player_action_list, player_money_list_round1, player_money_list_round2, community_card = self.action_history_player(history)

    # target_player_i :fold
    if player_action_list[target_player_i][-1] == "f":
      return -player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]

    #周りがfold
    last_play =[hi[-1] for idx, hi in enumerate(player_action_list) if idx != target_player_i]
    if last_play.count("f") == self.NUM_PLAYERS - 1:
      return sum(player_money_list_round1) + sum(player_money_list_round2) - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]

    #何人かでshow down
    show_down_player =[idx for idx, hi in enumerate(player_action_list) if hi[-1] != "f"]
    show_down_player_card = {}
    for idx in show_down_player:
      show_down_player_card[idx] = self.Rank(history[idx], community_card)
    max_rank = max(show_down_player_card.values())
    if show_down_player_card[target_player_i] != max_rank:
      return - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]
    else:
      win_num = len([idx for idx, card_rank in show_down_player_card.items() if card_rank == max_rank])
      return int((sum(player_money_list_round1) + sum(player_money_list_round2))/win_num) - player_money_list_round1[target_player_i] - player_money_list_round2[target_player_i]



  # whetther terminal_states
  def whether_terminal_states(self, history):
    """return string
    >>> LeducTrainer().whether_terminal_states("JKccKr")
    False
    >>> LeducTrainer().whether_terminal_states("QJccJcc")
    True
    >>> LeducTrainer().whether_terminal_states("QQcr")
    False
    >>> LeducTrainer(num_players=3).whether_terminal_states("QKTrff")
    True
    >>> LeducTrainer(num_players=3).whether_terminal_states("KKTcccQcrcrcc")
    True
    """
    if history.count("f") == self.NUM_PLAYERS -1 :
      return True

    if self.card_num_check(history) == self.NUM_PLAYERS +1 :
      private_cards, history_before, community_card, history_after = self.Split_history(history)
      if history_after.count("r") == 0 and history_after.count("c") == self.NUM_PLAYERS:
        return True

      if history.count("r") >=1 :
        idx = 0
        for i,hi in enumerate(history_after):
          if hi == "r":
            idx = i
        if history_after[idx+1:].count("c") == self.NUM_PLAYERS -1 :
          return True

    return False


  def card_num_check(self, history):
    """return string
    >>> LeducTrainer(num_players=3).card_num_check("JKTccc")
    3
    >>> LeducTrainer(num_players=2).card_num_check("KQcr")
    2
    """
    cards = self.card_distribution()
    count = 0
    for hi in history:
      if hi in cards:
        count += 1
    return count


  def whether_chance_node(self, history):
    """return string
    >>> LeducTrainer().whether_chance_node("JKcc")
    True
    >>> LeducTrainer().whether_chance_node("KQcr")
    False
    >>> LeducTrainer().whether_chance_node("")
    True
    >>> LeducTrainer(num_players=3).whether_chance_node("KQTcc")
    False
    """
    if history == "":
      return True

    if self.card_num_check(history) == self.NUM_PLAYERS :
      if history.count("r") == 0 and history.count("c") == self.NUM_PLAYERS:
        return True

      if history.count("r") >=1 :
        idx = 0
        for i,hi in enumerate(history):
          if hi == "r":
            idx = i
        if history[idx+1:].count("c") == self.NUM_PLAYERS -1 :
          return True

    return False


  #make node or get node
  def Get_information_set_node_or_create_it_if_nonexistant(self, infoSet):
    node = self.nodeMap.get(infoSet)
    if node == None:
      node = Node(self.NUM_ACTIONS, infoSet, self.NUM_PLAYERS)
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
    if self.card_num_check(history) == self.NUM_PLAYERS + 1:
      private_cards, history_before, community_card, history_after = self.Split_history(history)

    player = self.action_player(history)


    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      if len(history) == 0:
        self.cards = self.card_distribution()
        random.shuffle(self.cards)
        nextHistory = "".join(self.cards[:self.NUM_PLAYERS])
        return self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p_list)
      else:
        nextHistory = history + self.cards[self.NUM_PLAYERS]
        return self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p_list)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0
    node.Get_strategy_through_regret_matching()
    strategy = self.Get_strategy(node)

    for ai in node.possible_action:
      nextHistory = history + self.ACTION_DICT[ai]
      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.chance_sampling_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)
      nodeUtil += node.strategy[ai] * util_list[ai]

    if (not self.eval) and  player == target_player_i:
      for ai in node.possible_action:
        regret = util_list[ai] - nodeUtil

        p_exclude = 1
        for idx in range(self.NUM_PLAYERS):
          if idx != player:
            p_exclude *= p_list[idx]

        node.regretSum[ai] += p_exclude * regret
        node.strategySum[ai] += strategy[ai] * p_list[player]

    return nodeUtil


  #chance sampling CFR
  def vanilla_CFR(self, history, target_player_i, iteration_t, p_list):
    if self.card_num_check(history) == self.NUM_PLAYERS + 1:
      private_cards, history_before, community_card, history_after = self.Split_history(history)

    player = self.action_player(history)

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
      if len(history) == 0:
        cards = self.card_distribution()
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards, self.NUM_PLAYERS+1)]
        utility_sum = 0
        for cards_i in tqdm(cards_candicates):
          self.cards_i = cards_i
          nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
          utility_sum += (1/len(cards_candicates))* self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)
        return  utility_sum
      else:
        nextHistory = history + self.cards_i[self.NUM_PLAYERS]
        return self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()

    util_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
    nodeUtil = 0

    strategy = self.Get_strategy(node)

    for ai in node.possible_action:
      nextHistory = history + self.ACTION_DICT[ai]
      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = strategy[ai]

      util_list[ai] = self.vanilla_CFR(nextHistory, target_player_i, iteration_t, p_list * p_change)

      nodeUtil += strategy[ai] * util_list[ai]

    if (not self.eval) and  player == target_player_i:
      for ai in node.possible_action:
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
    if self.card_num_check(history) == self.NUM_PLAYERS + 1:
      private_cards, history_before, community_card, history_after = self.Split_history(history)

    player = self.action_player(history)

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i)

    elif self.whether_chance_node(history):
          if len(history) == 0:
            self.cards = self.card_distribution()
            random.shuffle(self.cards)
            nextHistory = "".join(self.cards[:self.NUM_PLAYERS])
            return self.external_sampling_MCCFR(nextHistory, target_player_i)
          else:
            nextHistory = history  + self.cards[self.NUM_PLAYERS]
            return self.external_sampling_MCCFR(nextHistory, target_player_i)


    infoSet = history[player]  + history[self.NUM_PLAYERS:]
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
  def outcome_sampling_MCCFR(self, history, target_player_i, iteration_t, p_list,s):
    if self.card_num_check(history) == self.NUM_PLAYERS + 1:
      private_cards, history_before, community_card, history_after = self.Split_history(history)

    player = self.action_player(history)

    if self.whether_terminal_states(history):
      return self.Return_payoff_for_terminal_states(history, target_player_i) / s, 1

    elif self.whether_chance_node(history):
          if len(history) == 0:
            self.cards = self.card_distribution()
            random.shuffle(self.cards)
            nextHistory = "".join(self.cards[:self.NUM_PLAYERS])
            return self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list, s)
          else:
            nextHistory = history + self.cards[self.NUM_PLAYERS]
            return self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list, s)

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    node.Get_strategy_through_regret_matching()
    probability =  np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)

    if player == target_player_i:
      for ai in node.possible_action:
        probability[ai] =  self.epsilon/len(node.possible_action)+ (1-self.epsilon)* node.strategy[ai]
    else:
      for ai in node.possible_action:
        probability[ai] = node.strategy[ai]

    #print(node.strategy, probability)
    sampling_action = np.random.choice(list(range(self.NUM_ACTIONS)), p=probability)
    nextHistory = history + self.ACTION_DICT[sampling_action]


    if player == target_player_i:

      p_change = np.array([1 for _ in range(self.NUM_PLAYERS)], dtype=float)
      p_change[player] = node.strategy[sampling_action]

      util, p_tail = self.outcome_sampling_MCCFR(nextHistory, target_player_i, iteration_t, p_list*p_change, s*probability[sampling_action])

      p_exclude = 1
      for idx in range(self.NUM_PLAYERS):
        if idx != player:
          p_exclude *= p_list[idx]

      w = util * p_exclude
      for ai in node.possible_action:
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

        for ai in node.possible_action:
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

      #if iteration_t in [int(j)-1 for j in np.logspace(1, len(str(self.train_iterations))-1, (len(str(self.train_iterations))-1)*3)] :
      #  self.exploitability_list[iteration_t] = self.get_exploitability_dfs()

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
      if self.card_num_check(history) == self.NUM_PLAYERS + 1:
        private_cards, history_before, community_card, history_after = self.Split_history(history)

      player = self.action_player(history)

      if self.whether_terminal_states(history):
        return self.Return_payoff_for_terminal_states(history, best_response_player)

      elif self.whether_chance_node(history):
        if len(history) == 0:
          cards = self.card_distribution()
          cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards, self.NUM_PLAYERS+1)]
          utility_sum = 0
          for cards_i in tqdm(cards_candicates):
            self.cards_i = cards_i
            nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
            utility_sum +=  (1/len(cards_candicates))* self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
          return utility_sum
        else:
          nextHistory = history + self.cards_i[self.NUM_PLAYERS]
          return self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)

      infoSet = history[player] + history[self.NUM_PLAYERS:]
      node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)

      if player == best_response_player:
        if infoSet not in best_response_strategy:
          action_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          br_value = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)

          for assume_history, po_ in self.infoSets_dict[infoSet]:
            for ai in node.possible_action:
              nextHistory = assume_history + self.ACTION_DICT[ai]
              br_value[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, po_)
              action_value[ai] += br_value[ai] * po_

          br_action = 0
          for ai in node.possible_action:
            if action_value[ai] > action_value[br_action]:
              br_action = ai
          best_response_strategy[infoSet] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
          best_response_strategy[infoSet][br_action] = 1.0

        node_util = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in node.possible_action:
          nextHistory = history + self.ACTION_DICT[ai]
          node_util[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob)
        best_response_util = 0
        for ai in node.possible_action:
          best_response_util += node_util[ai] * best_response_strategy[infoSet][ai]

        return best_response_util

      else:
        avg_strategy = node.Get_average_information_set_mixed_strategy()
        nodeUtil = 0
        action_value_list = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)
        for ai in node.possible_action:
          nextHistory = history + self.ACTION_DICT[ai]
          action_value_list[ai] = self.calc_best_response_value(best_response_strategy, best_response_player, nextHistory, prob*avg_strategy[ai])
          nodeUtil += avg_strategy[ai] * action_value_list[ai]
        return nodeUtil


  def create_infoSets(self, history, target_player, po):
    if self.card_num_check(history) == self.NUM_PLAYERS + 1:
      private_cards, history_before, community_card, history_after = self.Split_history(history)

    player = self.action_player(history)

    if self.whether_terminal_states(history):
      return

    elif self.whether_chance_node(history):
      if len(history) == 0:
        cards = self.card_distribution()
        cards_candicates = [cards_candicate for cards_candicate in itertools.permutations(cards, self.NUM_PLAYERS+1)]
        for cards_i in tqdm(cards_candicates):
          self.cards_i = cards_i
          nextHistory = "".join(cards_i[:self.NUM_PLAYERS])
          self.create_infoSets(nextHistory, target_player, po)
        return
      else:
        nextHistory = history + self.cards_i[self.NUM_PLAYERS]
        self.create_infoSets(nextHistory, target_player, po)
        return

    infoSet = history[player] + history[self.NUM_PLAYERS:]
    if player == target_player:
      if self.infoSets_dict.get(infoSet) is None:
        self.infoSets_dict[infoSet] = []
      self.infoSets_dict[infoSet].append((history, po))


    node = self.Get_information_set_node_or_create_it_if_nonexistant(infoSet)
    for ai in node.possible_action:
      nextHistory = history + self.ACTION_DICT[ai]
      if player == target_player:
        self.create_infoSets(nextHistory, target_player, po)
      else:
        actionProb = node.Get_average_information_set_mixed_strategy()[ai]
        self.create_infoSets(nextHistory, target_player, po*actionProb)


  def get_exploitability_dfs(self):

    # 各information setを作成 & reach_probabilityを計算
    self.infoSets_dict = {}
    for target_player in range(self.NUM_PLAYERS):
      self.create_infoSets("", target_player, 1.0)

    print(len(self.infoSets_dict))
    exploitability = 0
    best_response_strategy = {}
    for best_response_player_i in range(self.NUM_PLAYERS):
        print("best_response_player_i:", best_response_player_i)
        exploitability += self.calc_best_response_value(best_response_strategy, best_response_player_i, "", 1)

    if exploitability < 0:
      return 1e-7
    #assert exploitability >= 0
    return exploitability


#学習
"""
leduc_trainer = LeducTrainer(train_iterations=10**1, num_players=3)
#leduc_trainer.train("vanilla_CFR")
leduc_trainer.train("chance_sampling_CFR")
#leduc_trainer.train("external_sampling_MCCFR")
#leduc_trainer.train("outcome_sampling_MCCFR")

print("avg util:", leduc_trainer.eval_strategy(0))

pd.set_option('display.max_rows', None)
result_dict = {}
for key, value in sorted(leduc_trainer.nodeMap.items()):
  result_dict[key] = value.Get_average_information_set_mixed_strategy()

df = pd.DataFrame(result_dict.values(), index=result_dict.keys(), columns=["Fold", "Call", "Raise"])
df.index.name = "Node"
df

#print(df)

"""
print("")
# random strategy_profileのexploitability
#→0.9166666666666665
for i in range(3,4):
  kuhn_poker_agent = LeducTrainer(train_iterations=0, num_players=i)
  print("{}人対戦:".format(i), kuhn_poker_agent.get_exploitability_dfs())


doctest.testmod()