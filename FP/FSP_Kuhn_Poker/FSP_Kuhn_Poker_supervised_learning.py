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




class SupervisedLearning:
  def __init__(self):
    self.NUM_ACTIONS = 2


  def SL_train_AVG(self, memory, target_player, strategy):
    """return dict
    >>> avg_strategy = {'J': np.array([1., 0.]), \
      'Jb': np.array([1., 0.]),\
      'Jp': np.array([0.666666, 0.333334]),\
      'Jpb': np.array([1., 0.]),\
      'K': np.array([1., 0.]),\
      'Kb': np.array([0., 1.]),\
      'Kp': np.array([0., 1.]),\
      'Kpb': np.array([0., 1.]),\
      'Q': np.array([1., 0.]),\
      'Qb': np.array([0.666666, 0.333334]),\
      'Qp': np.array([1., 0.]),\
      'Qpb': np.array([0.666666, 0.333334])}
    >>> SupervisedLearning().SL_train_AVG(["JKpp"], 0, avg_strategy)
    {'J': array([1., 0.]), \
      'Jb': array([1., 0.]),\
      'Jp': array([0.666666, 0.333334]),\
      'Jpb': array([1., 0.]),\
      'K': array([1., 0.]),\
      'Kb': array([0., 1.]),\
      'Kp': array([0., 1.]),\
      'Kpb': array([0., 1.]),\
      'Q': array([1., 0.]),\
      'Qb': array([0.666666, 0.333334]),\
      'Qp': array([1., 0.]),\
      'Qpb': array([0.666666, 0.333334])}
    """

    update_strategy = copy.deepcopy(strategy)
    for node_X , _ in strategy.items():
      update_strategy[node_X] = np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)


    for one_episode in memory:
      one_episode_split = self.Episode_split(one_episode)

      for X, y in one_episode_split:
        if (target_player == 0 and len(X) %2 != 0) or (target_player == 1 and len(X) %2 == 0):
          if y == "p":
            update_strategy[X] += np.array([1.0, 0.0], dtype=float)
          else:
            update_strategy[X] += np.array([0.0, 1.0], dtype=float)


    for node_X , action_prob in update_strategy.items():
        if all(update_strategy[node_X] == np.array([0 for _ in range(self.NUM_ACTIONS)], dtype=float)):
          pass
        else:
          strategy[node_X] = update_strategy[node_X] / np.sum(action_prob)

    return strategy


  def SL_train_MLP(self, memory, update_strategy):
      #target_playerの戦略のみを更新する必要がある
      #SL_train_AVGを参考に

      train_X = np.array([])
      train_y = np.array([])
      for one_episode in memory:
        train = self.From_episode_to_bit(one_episode)
        for train_i in train:
          train_X = np.append(train_X, train_i[0])
          train_y = np.append(train_y, train_i[1])

      train_X = train_X.reshape(-1, 7)
      train_y = train_y.reshape(-1, 1)
      #print(train_X.shape, train_y.shape)

      #モデル構築 多層パーセプトロン
      clf = MLPClassifier()
      clf.fit(train_X, train_y)

      for node_X , _ in update_strategy.items():
        node_bit_X = self.make_X(node_X).reshape(-1, 7)
        y = clf.predict_proba(node_bit_X).ravel()
        update_strategy[node_X] = y


  def From_episode_to_bit(self, one_episode):
    """return list
    >>> SupervisedLearning().From_episode_to_bit('QKbp')
    [(array([0, 1, 0, 0, 0, 0, 0]), array([1])), (array([0, 0, 1, 0, 1, 0, 0]), array([0]))]
    """
    one_episode_split = self.Episode_split(one_episode)
    one_episode_bit = []

    for X, y in one_episode_split:
      y_bit = self.make_y(y)
      X_bit = self.make_X(X)
      one_episode_bit.append((X_bit, y_bit))

    return one_episode_bit

  def make_y(self, y):
    if y == "p":
      y_bit = np.array([0])
    else:
      y_bit = np.array([1])
    return y_bit

  def make_X(self, X):
    X_bit = np.array([0, 0, 0, 0, 0, 0, 0])
    if len(X) == 1:
      X_bit = self.first_bit(X[0], X_bit)
    elif len(X) == 2:
      X_bit = self.first_bit(X[0], X_bit)
      X_bit = self.second_bit(X[1], X_bit)
    elif len(X) == 3:
      X_bit = self.first_bit(X[0], X_bit)
      X_bit = self.second_bit(X[1], X_bit)
      X_bit = self.third_bit(X[2], X_bit)
    return X_bit

  def first_bit(self, X0, X_bit):
    if X0 == "J":
      X_bit[0] = 1
    elif X0 == "Q":
      X_bit[1] = 1
    elif X0 == "K":
      X_bit[2] = 1
    return X_bit

  def second_bit(self, X1, X_bit):
    if X1 == "p":
      X_bit[3] = 1
    elif X1 == "b":
      X_bit[4] = 1
    return X_bit

  def third_bit(self, X1, X_bit):
    if X1 == "p":
      X_bit[5] = 1
    elif X1 == "b":
      X_bit[6] = 1
    return X_bit

  def Episode_split(self, one_episode):
    """return list
    >>> SupervisedLearning().Episode_split('QKbp')
    [('Q', 'b'), ('Kb', 'p')]
    >>> SupervisedLearning().Episode_split('KQpbp')
    [('K', 'p'), ('Qp', 'b'), ('Kpb', 'p')]
    """
    one_episode_split = []
    my_card = one_episode[0]
    opp_card = one_episode[1]

    if len(one_episode) == 4:
      one_episode_split.append((my_card, one_episode[2]))
      one_episode_split.append((opp_card+one_episode[2], one_episode[3]))
    else:
      one_episode_split.append((my_card, one_episode[2]))
      one_episode_split.append((opp_card+one_episode[2], one_episode[3]))
      one_episode_split.append((my_card+one_episode[2:4], one_episode[4]))

    return one_episode_split

doctest.testmod()
