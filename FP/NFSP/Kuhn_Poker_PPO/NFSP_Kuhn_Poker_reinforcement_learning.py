
# _________________________________ Library _________________________________
from platform import node
from importlib_metadata import distribution
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
from torch.distributions import Categorical



import warnings
warnings.filterwarnings('ignore')
doctest.testmod()
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
  def __init__(self, train_iterations, num_players, hidden_units_num, lr, epochs, sampling_num, gamma, tau, update_frequency, loss_function, kuhn_trainer_for_rl, random_seed, device):
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
    self.random_seed = random_seed
    self.device = device
    self.save_count = [0 for _ in range(self.NUM_PLAYERS)]

    self.rl_algo = None

    self.kuhn_trainer.random_seed_fix(self.random_seed)



    self.deep_q_network = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num).to(self.device)
    self.deep_q_network_target = DQN(state_num = self.STATE_BIT_LEN, action_num = self.num_actions, hidden_units_num = self.hidden_units_num).to(self.device)

    self.deep_q_network_target.load_state_dict(self.deep_q_network.state_dict())


    self.optimizer = optim.SGD(self.deep_q_network.parameters(), lr=self.lr)
    self.loss_fn = loss_function

    self.update_count =  [0 for _ in range(self.NUM_PLAYERS)]


  def RL_learn(self, memory, target_player, update_strategy, k):

    self.deep_q_network.train()
    self.deep_q_network_target.eval()
    self.epsilon = 0.06/(k**0.5)


    total_loss = []
    # train
    for _ in range(self.epochs):
      samples = random.sample(memory, min(self.sampling_num, len(memory)))
      train_states = [sars[0] for sars in samples]
      train_actions = [sars[1] for sars in samples]
      train_rewards = [sars[2] for sars in samples]
      train_next_states = [sars[3] for sars in samples]
      train_done = [sars[4] for sars in samples]

      train_states = torch.tensor(train_states).float().reshape(-1,self.STATE_BIT_LEN).to(self.device)
      train_actions = torch.tensor(train_actions).float().reshape(-1,1).to(self.device)
      train_rewards = torch.tensor(train_rewards).float().reshape(-1,1).to(self.device)
      train_next_states = torch.tensor(train_next_states).float().reshape(-1,self.STATE_BIT_LEN).to(self.device)
      train_done = torch.tensor(train_done).float().reshape(-1,1).to(self.device)

      if self.rl_algo == "dqn":
        outputs = self.deep_q_network_target(train_next_states).detach().max(axis=1)[0].unsqueeze(1)

      #Double DQN
      elif self.rl_algo == "ddqn":
        not_target_nn_max_action = np.argmax(self.deep_q_network(train_next_states).detach(), axis=1).reshape(-1,1)

        outputs = self.deep_q_network_target(train_next_states).gather(1,not_target_nn_max_action.type(torch.int64)).detach()


      q_targets = train_rewards + (1 - train_done) * self.gamma * outputs


      q_now = self.deep_q_network(train_states)
      q_now_value = q_now.gather(1, train_actions.type(torch.int64))


      loss =   self.loss_fn(q_targets, q_now_value)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      total_loss.append(loss.item())

      self.update_count[target_player] += 1


      if self.update_count[target_player] % self.update_frequency ==  0 :
        self.deep_q_network_target.load_state_dict(self.deep_q_network.state_dict())


    if self.kuhn_trainer.wandb_save and self.save_count[target_player] % 10 == 0:
      wandb.log({'iteration': k, 'loss_rl_{}'.format(target_player):  np.mean(total_loss)})
    self.save_count[target_player] += 1




    #eval
    self.deep_q_network.eval()
    with torch.no_grad():
      for node_X , _ in update_strategy.items():
        if (len(node_X)-1) % self.NUM_PLAYERS == target_player :
          inputs_eval = torch.tensor(self.kuhn_trainer.make_state_bit(node_X)).float().reshape(-1,self.STATE_BIT_LEN).to(self.device)
          y = self.deep_q_network.forward(inputs_eval).to('cpu').detach().numpy()


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





# PPO
class ActorCriticNetwork(nn.Module):
  def __init__(self, state_num, action_num, hidden_units_num):
    super(ActorCriticNetwork, self).__init__()
    self.hidden_units_num = hidden_units_num
    self.fc1 = nn.Linear(state_num, self.hidden_units_num)
    self.fc2 = nn.Linear(self.hidden_units_num, action_num)
    self.fc3 = nn.Linear(self.hidden_units_num, 1)

    self.softmax = nn.Softmax()

  def actor_forward(self, x):
    h1 = F.leaky_relu(self.fc1(x))
    output = self.softmax(self.fc2(h1))
    return output

  def critic_forward(self, x):
    h1 = F.leaky_relu(self.fc1(x))
    output = self.fc3(h1)
    return output


  def make_action(self, node_X):
    action_probability = self.actor_forward(node_X)


    action_distribution = Categorical(action_probability)
    # 確率分布に従って、行動をサンプリング
    action = action_distribution.sample()
    # probability log

    action_log_probability = action_distribution.log_prob(action)

    return action, action_log_probability


  def evaluate_action(self, node_X, action):
    state_value = self.critic_forward(node_X)
    action_probability = self.actor_forward(node_X)
    action_distribution = Categorical(action_probability)
    action_log_probability = action_distribution.log_prob(action)
    distribution_entropy = action_distribution.entropy()

    return action_probability, torch.squeeze(state_value), distribution_entropy


class PPO:
  def __init__(self, num_players, hidden_units_num, sampling_num):
    self.NUM_PLAYERS = num_players
    self.num_actions = 2
    self.num_states = (self.NUM_PLAYERS + 1) + 2*(self.NUM_PLAYERS *2 - 2)
    self.hidden_units_num = hidden_units_num
    self.sampling_num = sampling_num


    self.current_policy = ActorCriticNetwork(self.num_states, self.num_actions, self.hidden_units_num)
    self.optimizer = optim.Adam(self.current_policy.parameters())

    self.old_policy = ActorCriticNetwork(self.num_states, self.num_actions, self.hidden_units_num)
    self.old_policy.load_state_dict(self.current_policy.state_dict())

    self.loss_function = nn.MSELoss()

    self.memory = RL_memory()


  def select_action(self, node_X):
    with torch.no_grad():

      action, action_log_prob = self.old_policy.make_action(node_X)

      return action


  def RL_learn(self, memory):
    pass


class RL_memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.log_probs = []
    self.rewards = []
    self.done = []




an = ActorCriticNetwork(state_num=7, action_num=2, hidden_units_num=32)
data = torch.Tensor(np.random.rand(10,7))
#print(an.make_action(data, "1"))
