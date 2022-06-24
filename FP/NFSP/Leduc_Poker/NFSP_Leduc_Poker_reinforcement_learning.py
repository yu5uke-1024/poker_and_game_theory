
# _________________________________ Library _________________________________


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
  def __init__(self, train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, gamma, tau, update_frequency, leduc_trainer_for_rl, random_seed):
    self.train_iterations = train_iterations
    self.NUM_PLAYERS = num_players
    self.num_actions = 3
    self.action_id = {"f":0, "c":1, "r":2}
    self.STATE_BIT_LEN = 2* ( (self.NUM_PLAYERS + 1) + 3*(self.NUM_PLAYERS *3 - 2) ) - 3
    self.hidden_units_num = hidden_units_num
    self.lr = lr
    self.epochs = epochs
    self.sampling_num = sampling_num
    self.gamma = gamma
    self.tau = tau
    self.update_frequency = update_frequency
    self.random_seed = random_seed

    self.leduc_trainer = leduc_trainer_for_rl
    self.card_rank  = self.leduc_trainer.card_rank
    self.infoset_action_player_dict = {}


    self.deep_q_network = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)
    self.deep_q_network_target = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num)

    self.leduc_trainer.random_seed_fix(self.random_seed)

    for target_param, param in zip(self.deep_q_network_target.parameters(), self.deep_q_network.parameters()):
        target_param.data.copy_(param.data)


    self.optimizer = optim.SGD(self.deep_q_network.parameters(), lr=self.lr)

    self.update_count =  [0 for _ in range(self.NUM_PLAYERS)]


  def RL_learn(self, memory, target_player, update_strategy, k):

    self.deep_q_network.train()
    self.deep_q_network_target.eval()
    self.epsilon = 0.06/(k**0.5)

    total_loss = []

    train_states = np.array([])
    train_actions = np.array([])
    train_rewards = np.array([])
    train_next_states = np.array([])
    train_done = np.array([])
    s_prime_array = np.array([])
    outputs = np.array([])


    for s, a, r, s_prime in memory:
      s_bit = self.leduc_trainer.make_state_bit(s)
      a_bit = self.leduc_trainer.make_action_bit(a)
      s_prime_bit = self.leduc_trainer.make_state_bit(s_prime)

      if s_prime == None:
        done = 1
      else:
        done = 0


      train_states = np.append(train_states, s_bit)
      train_actions = np.append(train_actions, a_bit)
      train_rewards = np.append(train_rewards, r)
      train_next_states = np.append(train_next_states, s_prime_bit)
      s_prime_array = np.append(s_prime_array, s_prime)
      train_done = np.append(train_done, done)


    train_states = torch.from_numpy(train_states).float().reshape(-1,self.STATE_BIT_LEN)
    train_actions = torch.from_numpy(train_actions).float().reshape(-1,1)
    train_rewards = torch.from_numpy(train_rewards).float().reshape(-1,1)
    train_next_states = torch.from_numpy(train_next_states).float().reshape(-1,self.STATE_BIT_LEN)
    train_done = torch.from_numpy(train_done).float().reshape(-1,1)


    outputs_all = self.deep_q_network_target(train_next_states).detach()


    for node_X, Q_value in zip(s_prime_array, outputs_all):


      if node_X == None:
        outputs = np.append(outputs, 0)
      else:
        action_list = self.leduc_trainer.node_possible_action[node_X]
        max_idx = action_list[0]


        for ai in action_list:
          if Q_value[ai] >= Q_value[max_idx]:
            max_idx = ai


        outputs = np.append(outputs, Q_value[max_idx])


    outputs = torch.from_numpy(outputs).float().unsqueeze(1)


    q_targets = train_rewards + (1 - train_done) * self.gamma * outputs

    #x: train_states, t: q_target, y:q_now_value

    train_dataset = torch.utils.data.TensorDataset(train_states, q_targets, train_actions)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.sampling_num, shuffle=True)

    for _ in range(self.epochs):

      for x, t, a in train_dataset_loader:

        q_now = self.deep_q_network(x)

        y = q_now.gather(1, a.type(torch.int64))

        loss = F.mse_loss(t, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss.append(loss.item())


    if self.update_count[target_player] % self.update_frequency ==  0 :
      self.parameter_update()


    #eval
    self.deep_q_network.eval()
    with torch.no_grad():
      for node_X , _ in update_strategy.items():
        if self.infoset_action_player_dict[node_X] == target_player :

          inputs_eval = torch.from_numpy(self.leduc_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN)
          y = self.deep_q_network.forward(inputs_eval).detach().numpy()


          if np.random.uniform() < self.epsilon:   # 探索(epsilonの確率で)
            action = np.random.choice(self.leduc_trainer.node_possible_action[node_X])
            update_strategy[node_X] = np.array([0 for _ in range(self.num_actions)], dtype=float)
            update_strategy[node_X][action] = 1.0

          else:
            action_list = self.leduc_trainer.node_possible_action[node_X]
            max_idx = action_list[0]

            for ai in action_list:
              if y[0][ai] >= y[0][max_idx]:
                max_idx = ai
            update_strategy[node_X] = np.array([0 for _ in range(self.num_actions)], dtype=float)
            update_strategy[node_X][max_idx] = 1.0



    #if k in [int(j) for j in np.logspace(0, len(str(self.train_iterations)), (len(str(self.train_iterations)))*4 , endpoint=False)] :
    if self.leduc_trainer.wandb_save:
      wandb.log({'iteration': k, 'loss_rl': np.mean(total_loss)})



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
