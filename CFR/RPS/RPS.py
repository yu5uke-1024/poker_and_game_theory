#Library
import random
from tqdm.notebook import tqdm


#RPS trainer
class RPSTrainer:
  def __init__(self, oppStrategy, iterations):
    self.ROCK = 0
    self.PAPER = 1
    self.SCISSORS = 2
    self.NUM_ACTIONS = 3

    self.regretSum = [0 for _ in range(self.NUM_ACTIONS)]
    self.strategy = [0 for _ in range(self.NUM_ACTIONS)]
    self.strategySum = [0 for _ in range(self.NUM_ACTIONS)]
    self.oppStrategy = oppStrategy
    self.iterations = iterations
    self.actionUtility = [0 for _ in range(self.NUM_ACTIONS)]


  def getStrategy(self):
    self.normalizingSum = 0
    for a in range(self.NUM_ACTIONS):
      self.strategy[a] = self.regretSum[a] if self.regretSum[a]>0 else 0
      self.normalizingSum += self.strategy[a]

    for a in range(self.NUM_ACTIONS):
      if self.normalizingSum >0 :
        self.strategy[a] /= self.normalizingSum
      else:
        self.strategy[a] = 1/self.NUM_ACTIONS
      self.strategySum[a] += self.strategy[a]

  def getAction(self, strategy):
    r = random.random()
    a = 0
    self.cumulativeProbability = 0
    while  a < self.NUM_ACTIONS -1 :
      self.cumulativeProbability += strategy[a]
      if r < self.cumulativeProbability:
        break
      a += 1
    return a


  def Get_regret_matched_mixed_strategy_actions(self):
    self.getStrategy()
    self.myAction = self.getAction(self.strategy)
    self.otherAction = self.getAction(self.oppStrategy)


  def Compute_action_utilities(self):
    self.actionUtility[self.otherAction] = 0
    self.actionUtility[0 if self.otherAction == self.NUM_ACTIONS -1 else self.otherAction+1] = 1
    self.actionUtility[self.NUM_ACTIONS-1 if self.otherAction == 0 else self.otherAction-1] = -1


  def Accumulate_action_regrets(self):
    for a in range(self.NUM_ACTIONS):
      self.regretSum[a] += self.actionUtility[a] - self.actionUtility[self.myAction]


  def train(self):
    for i in tqdm(range(self.iterations)):
      self.Get_regret_matched_mixed_strategy_actions()
      #print(self.regretSum, self.strategy, self.myAction, self.otherAction)
      self.Compute_action_utilities()
      self.Accumulate_action_regrets()


  def Get_average_mixed_strategy(self):
    self.avgStrategy = [0 for _ in range(self.NUM_ACTIONS)]
    self.normalizingSum = 0
    for a in range(self.NUM_ACTIONS):
      self.normalizingSum += self.strategySum[a]
    for a in range(self.NUM_ACTIONS):
      if self.normalizingSum >0 :
        self.avgStrategy[a] = self.strategySum[a] / self.normalizingSum
      else:
        self.avgStrategy[a] = 1/ self.NUM_ACTIONS
    return self.avgStrategy

#config
oppStrategy = [0.4, 0.3, 0.3]
iterations = 100000


#train
trainer = RPSTrainer(oppStrategy=oppStrategy, iterations=iterations)
trainer.train()


#result
print("opponent strategy:", trainer.oppStrategy)
print("my optimal strategy:", trainer.Get_average_mixed_strategy())
