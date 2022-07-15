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


class Agent:
    def __init__(self, state_num, action_num, hidden_unit_num, epochs, batch_size, gamma, lam, policy_clip, lr_actor, lr_critic):
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_unit_num = hidden_unit_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.policy_clip = policy_clip
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.MseLoss = nn.MSELoss()
        self.memory = PPOMemory(self.batch_size)


        self.policy = ActorCriticNetwork(self.state_num, self.action_num, self.hidden_unit_num)



        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])

        self.policy_old = ActorCriticNetwork(self.state_num, self.action_num, self.hidden_unit_num)
        self.policy_old.load_state_dict(self.policy.state_dict())



    def choose_action(self, observation):
        with torch.no_grad():
            state = torch.tensor(np.array(observation),dtype=torch.float)

            action, log_prob = self.policy_old.act(state)


        log_prob = torch.squeeze(log_prob).item()
        action = torch.squeeze(action).item()

        return action, log_prob



    def save_memory(self, state, action, log_prob, reward, done):
        self.memory.store(state, action, log_prob, reward, done)


    def learn(self):
        for _ in range(self.epochs):
            state_arr, action_arr, old_action_prob_arr, reward_arr,  done_arr, batch_index = self.memory.get_batch()

            action_prob_arr, value_arr, dist_entropy =  self.policy.evaluate(torch.tensor(state_arr, dtype=torch.float), torch.tensor(action_arr, dtype=torch.float))


            #calulate adavantage (一旦、愚直に前から計算する)
            advantages = np.zeros(len(state_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1 # lamda * gamma が 足されていってるもの
                adavantage_t = 0
                for k in range(t, len(reward_arr)-1):
                    sigma_k = reward_arr[k] + self.gamma * value_arr[k+1] * (1 - done_arr[1]) - value_arr[k]
                    adavantage_t += discount * sigma_k
                    discount += self.gamma * self.lam
                advantages[t] = adavantage_t


            advantages = torch.tensor(advantages)

            #batch 計算
            for batch in batch_index:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                actions = torch.tensor(action_arr[batch])
                old_action_probs = torch.tensor(old_action_prob_arr[batch])

                action_probs = torch.tensor(action_prob_arr[batch])


                #元々log probだったから * expで 方策に戻る  policy_ratio: 現在とひとつ前の方策を比較
                policy_ratio =  action_probs.exp() / old_action_probs.exp()

                #candicate 1
                weighted_probs = policy_ratio * advantages[batch]

                #candicate 2  第一引数inputに処理する配列Tesonr、第二引数minに最小値、第三引数maxに最大値を指定
                weighted_clipped_probs = torch.clamp(policy_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages[batch]

                actor_loss = torch.min(weighted_probs, weighted_clipped_probs)


                # calculate critic loss
                critic_value = value_arr[batch]
                returns = advantages[batch] * value_arr[batch]

                #total loss
                #total_loss = - actor_loss + 0.5 * self.MseLoss(critic_value, returns)
                total_loss = 0.5 * self.MseLoss(critic_value, returns)



                self.optimizer.zero_grad()
                total_loss.mean().backward()
                self.optimizer.step()


        self.memory.delete()







class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def delete(self):
        self.states = []
        self.actions = []
        self.log_probs = []
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
             np.array(self.rewards), np.array(self.dones), batch_index



class ActorCriticNetwork(nn.Module):
    def __init__(self, state_num, action_num, hidden_units_num):
        super(ActorCriticNetwork, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
        self.hidden_units_num = hidden_units_num


        self.actor = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, self.action_num),
            nn.Softmax(dim = 0)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_units_num),
            nn.ReLU(),
            nn.Linear(self.hidden_units_num, 1)
        )


    def act(self, x):
        act_prob = self.actor(x)
        dist = Categorical(act_prob)

        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.detach(), action_log_prob.detach()


    def evaluate(self, x, a):
        action_prob = self.actor(x)
        dist = Categorical(action_prob)
        action_log_probs = dist.log_prob(a)
        dist_entorpy = dist.entropy()
        values = self.critic(x)

        return action_log_probs, values, dist_entorpy




class PPO:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.episode = 0
        self.iteration = 5
        self.number_parallel = 10
        self.hidden_unit_num = 128
        self.batch_size = 64
        self.epochs = 1
        self.gamma = 0.9
        self.lam = 1
        self.policy_clip = 0.2
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
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
            lr_actor= self.lr_actor,
            lr_critic= self.lr_critic,
            )




    def train(self):

        for _ in tqdm(range(self.iteration)):

            # collect trajectory
            for _ in range(self.number_parallel):
                score = 0
                observation = self.env.reset()
                done = False

                # start 1 episode
                while not done:

                    action, log_prob = self.agent.choose_action(observation)
                    next_observation, reward, done, info = self.env.step(action)

                    score += 1

                    self.agent.save_memory(observation, action, log_prob, reward, done)

                    next_observation = observation

            self.history_score.append(score)
            self.avg_score = np.mean(self.history_score[-100:])
            self.hisotry_avg_score.append(self.avg_score)



            # 学習して、良い方策へ
            self.agent.learn()



PPO_trainer = PPO()
PPO_trainer.train()
#print(PPO_trainer.history_score)
#print(PPO_trainer.hisotry_avg_score)

#plt.plot(range(len(PPO_trainer.hisotry_avg_score)), PPO_trainer.hisotry_avg_score)
#plt.xlabel("iteration")
#plt.ylabel("avg score")
#plt.show()
