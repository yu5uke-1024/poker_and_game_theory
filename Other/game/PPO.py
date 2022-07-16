# -- library --
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#https://pytorch.org/docs/stable/distributions.html
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt


class PPO:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.episode = 0
        self.iteration = 100
        self.number_parallel = 5
        self.hidden_unit_num = 256
        self.batch_size = 5
        self.epochs = 4
        self.gamma = 0.99
        self.lam = 0.95
        self.policy_clip = 0.2
        self.policy_lr = 0.0003
        self.value_lr = 0.001
        self.N = 20
        self.history_score = []
        self.hisotry_avg_score = []


        self.agent = Agent(
            state_num= self.env.observation_space.shape[0],
            action_num= self.env.action_space.n,
            hidden_unit_num= self.hidden_unit_num,
            batch_size= self.batch_size,
            epochs= self.epochs,
            gamma= self.gamma,
            lam= self.lam,
            policy_clip= self.policy_clip,
            policy_lr = self.policy_lr,
            value_lr = self.value_lr
            )




    def train(self):
        n_step = 0
        learn_iters = 0

        for i in tqdm(range(self.iteration)):

            # collect trajectory
            for _ in range(self.number_parallel):
                observation = self.env.reset()
                score = 0
                done = False

                # start 1 episode
                while not done:

                    action, log_prob, value = self.agent.choose_action(observation)
                    next_observation, reward, done, info = self.env.step(action)


                    score += reward
                    n_step += 1

                    self.agent.save_memory(observation, action, log_prob, value, reward, done)


                    observation = next_observation

                self.history_score.append(score)
                self.avg_score = np.mean(self.history_score[-100:])
                self.hisotry_avg_score.append(self.avg_score)


            # 学習して、良い方策へ
            self.agent.learn()
            learn_iters += 1


            if i % 10 == 0:
                print(i, self.avg_score , n_step, learn_iters)


class Agent:
    def __init__(self, state_num, action_num, hidden_unit_num, epochs, batch_size, gamma, lam, policy_clip, policy_lr, value_lr):
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_unit_num = hidden_unit_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.policy_clip = policy_clip
        self.policy_lr = policy_lr
        self.value_lr = value_lr

        self.actor = ActorNetwork(self.state_num, self.action_num, self.hidden_unit_num, self.policy_lr)
        self.critic = CriticNetwork(self.state_num, self.hidden_unit_num, self.value_lr)
        self.memory = PPOMemory(self.batch_size)


    def choose_action(self, observation):

        state = torch.tensor(np.array(observation),dtype=torch.float)

        dist = self.actor(state)

        value = self.critic(state)

        action = dist.sample()

        #ただの数字を取り出す
        log_prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, log_prob, value


    def save_memory(self, state, action, log_prob, value, reward, done):
        self.memory.store(state, action, log_prob, value, reward, done)


    def calculate_advantage(self, state_arr, reward_arr, value_arr, done_arr):

        #calulate adavantage (一旦、愚直に前から計算する)
        advantages = np.zeros(len(state_arr), dtype=np.float32)

        for t in range(len(reward_arr)-1):
            discount = 1 # lamda * gamma が 足されていってるもの
            adavantage_t = 0
            for k in range(t, len(reward_arr)-1):
                sigma_k = reward_arr[k] + self.gamma * value_arr[k+1] * (1 - done_arr[k]) - value_arr[k]
                adavantage_t += discount * sigma_k
                discount *= self.gamma * self.lam

            advantages[t] = adavantage_t

        return advantages


    def learn(self):
        for _ in range(self.epochs):
            state_arr, action_arr, old_action_prob_arr, value_arr, reward_arr,  done_arr, batch_index = self.memory.get_batch()

            advantages = self.calculate_advantage(state_arr, reward_arr, value_arr, done_arr)

            advantages = torch.tensor(advantages)
            values = torch.tensor(value_arr)


            #batch 計算
            for batch in batch_index:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                actions = torch.tensor(action_arr[batch])
                old_action_probs = torch.tensor(old_action_prob_arr[batch])


                # calculate actor loss
                dist = self.actor(states)
                new_action_probs = dist.log_prob(actions)

                #元々log probだったから * expで 方策に戻る  policy_ratio: 現在とひとつ前の方策を比較
                policy_ratio =  new_action_probs.exp() / old_action_probs.exp()

                #candicate 1
                weighted_probs = policy_ratio * advantages[batch]

                #candicate 2  第一引数inputに処理する配列Tesonr、第二引数minに最小値、第三引数maxに最大値を指定
                weighted_clipped_probs = torch.clamp(policy_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages[batch]

                actor_loss = - torch.min(weighted_probs, weighted_clipped_probs).mean()


                # calculate critic loss
                critic_value = torch.squeeze(self.critic(states))

                returns = advantages[batch] + values[batch]

                critic_loss = ((returns-critic_value)**2).mean()


                #total loss
                total_loss = actor_loss + 0.5 * critic_loss


                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()


        self.memory.delete()





class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def delete(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


    def get_size(self):
        return len(self.states)


    def get_batch(self):
        #dataをbatch_sizeに切り分ける そのindexを作る
        size = self.get_size()
        batch_start_index = np.arange(0, size, self.batch_size)
        state_index = np.arange(size, dtype=np.int64)
        np.random.shuffle(state_index)
        batch_index = [state_index[i:i+self.batch_size] for i in batch_start_index]


        #npだと array[batch] で 取得可能
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
            np.array(self.values), np.array(self.rewards), np.array(self.dones), batch_index






class CriticNetwork(nn.Module):
    def __init__(self, state_num, hidden_units_num, lr):
        super(CriticNetwork, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num
        self.lr = lr

        self.critic = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, 1)
            )

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)


    def forward(self, x):
        output = self.critic(x)
        return output


class ActorNetwork(nn.Module):
    def __init__(self, state_num, action_num, hidden_units_num, lr):
        super(ActorNetwork, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_units_num = hidden_units_num
        self.lr = lr

        self.actor = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.action_num),
            nn.Softmax(dim = -1)
            )

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x):
        h1 = self.actor(x)
        dist = Categorical(h1)

        #action distribution
        return dist






PPO_trainer = PPO()
PPO_trainer.train()
#print(PPO_trainer.history_score)
#print(PPO_trainer.hisotry_avg_score)

plt.plot(range(len(PPO_trainer.hisotry_avg_score)), PPO_trainer.history_score)
plt.xlabel("iteration")
plt.ylabel("avg score")
plt.show()
