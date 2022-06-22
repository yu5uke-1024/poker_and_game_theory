
# _________________________________ Library _________________________________
from platform import node
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import doctest
from collections import deque
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

# _________________________________ RL NN class _________________________________
class DQN(nn.Module):

    def __init__(self, state_num, action_num, hidden_units_num):
        super(DQN, self).__init__()
        self.hidden_units_num = hidden_units_num
        self.fc1 = nn.Linear(state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, action_num)

    def forward(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        output = self.fc2(h1)
        return output


# _________________________________ RL class _________________________________
class ReinforcementLearning:
  def __init__(self, train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, gamma, tau, update_frequency, loss_function, kuhn_trainer_for_rl):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.num_actions = 2
    self.action_id = {"p":0, "b":1}
    self.STATE_BIT_LEN = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num
    self.gamma = gamma
    self.tau = tau
    self.update_frequency = update_frequency
    self.kuhn_trainer = kuhn_trainer_for_rl
    self.card_rank  = self.kuhn_trainer.card_rank


    self.deep_q_network = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)
    self.deep_q_network_target = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)


    for target_param, param in zip(self.deep_q_network_target.parameters(), self.deep_q_network.parameters()):
        target_param.data.copy_(param.data)


    self.optimizer = optim.SGD(self.deep_q_network.parameters(), lr=self.lr)
    self.loss_fn = loss_function


  def RL_learn(self, memory, target_player, update_strategy, k):

    self.deep_q_network.train()
    self.deep_q_network_target.eval()
    self.epsilon = 0.06/(k**0.5)


    total_loss = 0
    # train
    for _ in range(self.epochs):

      samples = random.sample(memory, min(self.sampling_num, len(memory)))


      train_states = np.array([])
      train_actions = np.array([])
      train_rewards = np.array([])
      train_next_states = np.array([])
      train_done = np.array([])


      for s, a, r, s_prime in samples:
        s_bit = self.kuhn_trainer.make_state_bit(s)
        a_bit = self.kuhn_trainer.make_action_bit(a)
        s_prime_bit = self.kuhn_trainer.make_state_bit(s_prime)
        if s_prime == None:
          done = 1
        else:
          done = 0



        train_states = np.append(train_states, s_bit)
        train_actions = np.append(train_actions, a_bit)
        train_rewards = np.append(train_rewards, r)
        train_next_states = np.append(train_next_states, s_prime_bit)
        train_done = np.append(train_done, done)

      train_states = torch.from_numpy(train_states).float().reshape(-1,self.STATE_BIT_LEN)
      train_actions = torch.from_numpy(train_actions).float().reshape(-1,1)
      train_rewards = torch.from_numpy(train_rewards).float().reshape(-1,1)
      train_next_states = torch.from_numpy(train_next_states).float().reshape(-1,self.STATE_BIT_LEN)
      train_done = torch.from_numpy(train_done).float().reshape(-1,1)

      outputs = self.deep_q_network_target(train_next_states).detach().max(axis=1)[0].unsqueeze(1)


      q_targets = train_rewards + (1 - train_done) * self.gamma * outputs

      q_now = self.deep_q_network(train_states)
      q_now = q_now.gather(1, train_actions.type(torch.int64))


      loss =   self.loss_fn(q_targets, q_now)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      total_loss += loss.item()


    if k % self.update_frequency ==  0:
      self.parameter_update()


    if k in [int(j) for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*4 , endpoint=False)] :
      if self.kuhn_trainer.wandb_save:
        wandb.log({'iteration': k, 'loss_rl': total_loss/self.epochs})

      """
      self.deep_q_network.eval()
      with torch.no_grad():
        for node_X , _ in update_strategy.items():
          if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
            inputs_eval = torch.from_numpy(self.kuhn_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)
            y = self.deep_q_network.forward(inputs_eval).detach().numpy()
            print(node_X, y)
      print("")
      """


    #eval
    self.deep_q_network.eval()
    with torch.no_grad():
      for node_X , _ in update_strategy.items():
        if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
          inputs_eval = torch.from_numpy(self.kuhn_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)
          y = self.deep_q_network.forward(inputs_eval).detach().numpy()


          if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
            action = np.random.randint(self.num_actions)
            if action == 0:
              update_strategy[node_X] = np.array([1, 0], dtype=float)
            else:
              update_strategy[node_X] = np.array([0, 1], dtype=float)

          else:
            if y[0][0] > y[0][1]:
              update_strategy[node_X] = np.array([1, 0], dtype=float)
            else:
              update_strategy[node_X] = np.array([0, 1], dtype=float)


          #print(node_X, y)



  def parameter_update(self):

    # soft update
    """
    for target_param, param in zip(self.deep_q_network_target.parameters(), self.deep_q_network.parameters()):
      target_param.data.copy_(
          self.tau * param.data + (1.0 - self.tau) * target_param.data)
    """

    # hard update
    for target_param, param in zip(self.deep_q_network_target.parameters(), self.deep_q_network.parameters()):
      target_param.data.copy_(param.data)




doctest.testmod()
