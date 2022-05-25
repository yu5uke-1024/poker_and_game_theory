#Library
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

import warnings
warnings.filterwarnings('ignore')

import FSP_Leduc_Poker_trainer



class SupervisedLearning:
  def __init__(self, num_players=2, num_actions=2, node_possible_action=None, infoset_action_player_dict=None):
    self.num_players = num_players
    self.num_actions = num_actions
    self.node_possible_action = node_possible_action
    self.max_len_X_bit = 2* ( (self.num_players + 1) + 3*(self.num_players *3 - 2) )
    self.ACTION_DICT = {0:"f", 1:"c", 2:"r"}
    self.ACTION_DICT_verse = {"f":0, "c":1, "r":2}
    self.infoset_action_player_dict = infoset_action_player_dict


    self.cards = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    self.card_order  = self.make_card_order(self.num_players)
    self.leduc_trainer = FSP_Leduc_Poker_trainer.LeducTrainer(num_players=self.num_players)




  def make_card_order(self, num_players):
    """return dict
    >>> SupervisedLearning().make_card_order(2) == {'J':0, 'Q':1, 'K':2}
    True
    >>> SupervisedLearning().make_card_order(3) == {'T':0, 'J':1, 'Q':2, 'K':3}
    True
    """
    card_order = {}
    for i in range(num_players+1):
      card_order[self.cards[11-num_players+i]] =  i

    return card_order


  def SL_train_AVG(self, memory, target_player, strategy, n_count):
    for one_episode in memory:
      one_episode_split = self.Episode_split(one_episode)

      for X, y in one_episode_split:
        if self.infoset_action_player_dict[X] == target_player :
          action_prob_list = np.array([0 for _ in range(self.num_actions)], dtype=float)
          action_prob_list[self.ACTION_DICT_verse[y]] = 1.0
          n_count[X] += action_prob_list


    for node_X , action_prob in n_count.items():
        strategy[node_X] = n_count[node_X] / np.sum(action_prob)

    return strategy



  def SL_train_MLP(self, memory, target_player, update_strategy):

      train_X = np.array([])
      train_y = np.array([])
      for one_episode in memory:
        train = self.From_episode_to_bit(one_episode, target_player)
        for train_i in train:
          train_X = np.append(train_X, train_i[0])
          train_y = np.append(train_y, train_i[1])


      #最初にtrain_y に 0,1,2全てそろわない可能性あり　3値分類でなくなってしまう
      for action_id in range(self.num_actions):
        train_X = np.append(train_X, [0 for _ in range(self.max_len_X_bit)])
        train_y = np.append(train_y, [action_id])



      train_X = train_X.reshape(-1, self.max_len_X_bit)
      train_y = train_y.reshape(-1, 1)




      #モデル構築 多層パーセプトロン
      clf = MLPClassifier(hidden_layer_sizes=(200,))
      clf.fit(train_X, train_y)

      for node_X , _ in update_strategy.items():
        node_bit_X = self.make_X(node_X).reshape(-1, self.max_len_X_bit)
        y = clf.predict_proba(node_bit_X).ravel()

        #print("first:", y)
        possible_action_list = self.node_possible_action[node_X]
        normalizationSum = 0
        for action_i, yi in enumerate(y):
          if action_i not in possible_action_list:
            y[action_i] = 0
          else:
            normalizationSum += yi

        y /= normalizationSum
        update_strategy[node_X] = y



  def From_episode_to_bit(self, one_episode, target_player):
    """return list
    >>> SupervisedLearning(2, 2).From_episode_to_bit('QKrf', 0)
    [(array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0]), array([2]))]
    >>> SupervisedLearning(2, 2).From_episode_to_bit('QKccJcc', 0)
    [(array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0]), array([1])), (array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1]), array([1]))]
    """
    one_episode_split = self.Episode_split(one_episode)
    one_episode_bit = []
    for X, y in one_episode_split:
      if self.infoset_action_player_dict[X] == target_player :

        y_bit = self.make_y(y)
        X_bit = self.make_X(X)
        one_episode_bit.append((X_bit, y_bit))

    return one_episode_bit


  def make_y(self, y):
    if y == "f":
      y_bit = np.array([0])
    elif y == "c":
      y_bit = np.array([1])
    elif y == "r":
      y_bit = np.array([2])
    return y_bit


  # com card X_bit の最後のn+1を使う
  def make_X(self, X):

    X_bit = np.array([0 for _ in range(self.max_len_X_bit)])
    X_bit[self.card_order[X[0]]] = 1

    for idx, Xi in enumerate(X[1:]):
      if Xi not in self.cards:
        X_bit[(self.num_players+1) + 3*idx + self.ACTION_DICT_verse[Xi]] = 1
      else:
        com_idx = self.card_order[Xi] + 1
        X_bit[-com_idx] = 1

    return X_bit


  def Episode_split(self, one_episode):
    """return list
    >>> SupervisedLearning(2, 3).Episode_split('QKccJcc')
    [('Q', 'c'), ('Kc', 'c'), ('QccJ', 'c'), ('KccJc', 'c')]
    >>> SupervisedLearning(3, 3).Episode_split('QKTcrfcJrf')
    [('Q', 'c'), ('Kc', 'r'), ('Tcr', 'f'), ('Qcrf', 'c'), ('QcrfcJ', 'r'), ('KcrfcJr', 'f')]
    """
    one_episode_split = []

    player_card = one_episode[:self.num_players]
    action_history = one_episode[self.num_players:]
    act = ""

    for ai in action_history:
      if ai not in self.cards:
        si = one_episode[self.leduc_trainer.action_player(player_card+act)] + act
        one_episode_split.append((si,ai))

      act += ai

    return one_episode_split

doctest.testmod()
