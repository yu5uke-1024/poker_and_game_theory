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
import math

import FSP_Leduc_Poker_trainer



class ReinforcementLearning:
  def __init__(self,random_seed, infoSet_dict_player, num_players, num_actions, node_possible_action=None, infoset_action_player_dict=None):
    self.gamma = 1
    self.num_players = num_players
    self.num_actions = num_actions
    self.ACTION_DICT_verse =  {"f":0, "c":1, "r":2}
    self.player_q_state = self.make_each_player_state_idx(infoSet_dict_player)
    self.leduc_trainer = FSP_Leduc_Poker_trainer.LeducTrainer(num_players=self.num_players)
    self.cards = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    self.node_possible_action = node_possible_action
    self.infoset_action_player_dict = infoset_action_player_dict
    self.random_seed = random_seed

    self.leduc_trainer.random_seed_fix(self.random_seed)



  def make_each_player_state_idx(self, infoSet_dict_player):
    #Q-state for each player
    player_q_state = [{} for _ in range(self.num_players)]
    for player_i, player_i_state in enumerate(infoSet_dict_player):
      for idx, j in enumerate(player_i_state):
        player_q_state[player_i][j] = idx
    return player_q_state


  def Episode_split(self, one_episode):
    """return list
    >>> ReinforcementLearning([],2, 2).Episode_split('QKccJcc')
    [('Q', 'c', 0, 'QccJ'), ('Kc', 'c', 0, 'KccJc'), ('QccJ', 'c', -1, None), ('KccJc', 'c', 1, None)]
    >>> ReinforcementLearning([],2, 2).Episode_split('QKrf')
    [('Q', 'r', 1, None), ('Kr', 'f', -1, None)]
    """
    one_episode_split = []

    player_card = one_episode[:self.num_players]
    action_history = one_episode[self.num_players:]
    act = one_episode[self.num_players:-1]
    player_last_infoset_list = [None for _ in range(self.num_players)]

    for ai in action_history[::-1]:
      if ai not in self.cards:
        player_i = self.leduc_trainer.action_player(player_card+act)
        si = one_episode[player_i] + act
        if player_last_infoset_list[player_i] == None:
          s_prime = None
          r = self.leduc_trainer.Return_payoff_for_terminal_states(one_episode, player_i)
          player_last_infoset_list[player_i] = si
        else:
          s_prime = player_last_infoset_list[player_i]
          player_last_infoset_list[player_i] = si
          r = 0


        one_episode_split.append((si, ai, r, s_prime))

      act = act[:-1]
    return one_episode_split[::-1]


  def RL_train(self, memory, target_player, update_strategy, q_value, k, rl_algo):
    self.alpha = 0.05/ (1+0.003*(k**0.5))
    self.T = 1 / (1+ 0.02*(k**0.5))
    self.epsilon = 0.6/(k**0.5)
    self.epochs = 1
    self.sample_num = 30

    for _ in range(self.epochs):
      if len(memory) <= self.sample_num:
        return

      replay_sample_list = random.sample(memory, self.sample_num)

      for one_episode in replay_sample_list:

        one_episode_split = self.Episode_split(one_episode)
        for trainsition in one_episode_split:
          s, a, r, s_prime = trainsition[0], trainsition[1], trainsition[2], trainsition[3]
          if self.infoset_action_player_dict[s] == target_player :
            s_idx = self.player_q_state[target_player][s]
            a_idx = self.ACTION_DICT_verse[a]

            if s_prime == None:
              q_value[s_idx][a_idx] = q_value[s_idx][a_idx]  + self.alpha*(r - q_value[s_idx][a_idx])

            else:
              s_prime_idx = self.player_q_state[target_player][s_prime]

              q_value_s__prime_max = q_value[s_prime_idx][self.node_possible_action[s_prime][0]]
              for ai in self.node_possible_action[s_prime]:
                if q_value_s__prime_max <= q_value[s_prime_idx][ai]:
                  q_value_s__prime_max = q_value[s_prime_idx][ai]

              q_value[s_idx][a_idx] = q_value[s_idx][a_idx]  + self.alpha*(r + self.gamma*q_value_s__prime_max- q_value[s_idx][a_idx])




    state_space = len(self.player_q_state[target_player])
    action_space = len(self.ACTION_DICT_verse)


    if rl_algo == "boltzmann":
      q_value_boltzmann = np.zeros((state_space,action_space))
      for si in range(state_space):
        for ai in range(action_space):
          q_value_boltzmann[si][ai] = math.exp(q_value[si][ai]/self.T)

      for state, idx in self.player_q_state[target_player].items():

        update_strategy[state] = q_value_boltzmann[self.player_q_state[target_player][state]] / sum(q_value_boltzmann[self.player_q_state[target_player][state]])
        normalizationSum = 0
        possible_action_list = self.node_possible_action[state]

        for action_i, yi in enumerate(update_strategy[state]):
          if action_i not in possible_action_list:
            update_strategy[state][action_i] = 0
          else:
            normalizationSum += yi

        assert normalizationSum != 0
        update_strategy[state] /= normalizationSum



    elif rl_algo == "epsilon-greedy":
      for state, idx in self.player_q_state[target_player].items():
        if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
          action = np.random.choice(self.node_possible_action[state])
          update_strategy[state] = np.array([0 for _ in range(self.num_actions)], dtype=float)
          update_strategy[state][action] = 1.0

        else:
          action_list = self.node_possible_action[state]
          state_id = self.player_q_state[target_player][state]
          max_idx = action_list[0]
          for ai in action_list:
            if q_value[state_id][ai] > q_value[state_id][max_idx]:
              max_idx = ai
          update_strategy[state] = np.array([0 for _ in range(self.num_actions)], dtype=float)
          update_strategy[state][max_idx] = 1.0


doctest.testmod()
