#ライブラリ
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import doctest


#Game trainer
class GameTrainer:
  def __init__(self, iterations):
    self.iterations = iterations
    self.avg_me_strategy = np.array([0.1, 0.9], dtype = "float")
    self.avg_opp_strategy = np.array([0.6, 0.4], dtype = "float")
    self.payoff = [[1,-1], [-1, 1], [-1,1], [1, -1]]


  def return_strategy_for_utility(self, action_0_utility, action_1_utility):
    """return vector
    >>> GameTrainer(iterations=100).return_strategy_for_utility(2, 4)
    array([0, 1])
    >>> GameTrainer(iterations=100).return_strategy_for_utility(1, -1)
    array([1, 0])
    """
    strategy = np.array([0, 0])

    if action_0_utility > action_1_utility:
      strategy[0] = 1
    else:
      strategy[1] = 1
    return strategy

  def calculate_best_response_startegy_for_me(self, opp_strategy):
    """return vector
    >>> GameTrainer(iterations=100).calculate_best_response_startegy_for_me(np.array([0.8,0.2]))
    array([1, 0])
    """
    action_0_utility =  opp_strategy[0]*self.payoff[0][0] + opp_strategy[1]*self.payoff[1][0]
    action_1_utility =  opp_strategy[0]*self.payoff[2][0] + opp_strategy[1]*self.payoff[3][0]
    return self.return_strategy_for_utility(action_0_utility, action_1_utility)

  def calculate_best_response_startegy_for_opp(self, me_strategy):
    action_0_utility =  me_strategy[0]*self.payoff[0][1] + me_strategy[1]*self.payoff[2][1]
    action_1_utility =  me_strategy[0]*self.payoff[1][1] + me_strategy[1]*self.payoff[3][1]
    return self.return_strategy_for_utility(action_0_utility, action_1_utility)


  def train(self):
    for iteration in tqdm(range(self.iterations)):
      alpha = 1/(iteration+2)

      best_response_for_me = self.calculate_best_response_startegy_for_me(self.avg_opp_strategy)
      best_response_for_opp = self.calculate_best_response_startegy_for_opp(self.avg_me_strategy)

      self.avg_me_strategy = (1-alpha)*self.avg_me_strategy + alpha*best_response_for_me
      self.avg_opp_strategy = (1-alpha)*self.avg_opp_strategy + alpha*best_response_for_opp

    return self.avg_me_strategy, self.avg_opp_strategy


#RPS Trainer
class RPSTrainer:
  def __init__(self, iterations):
    self.iterations = iterations
    self.avg_me_strategy = np.array([0.4, 0.4, 0.2], dtype = "float")
    self.avg_opp_strategy = np.array([0.0, 0.3, 0.7], dtype = "float")


  def calculate_best_response_startegy(self, other_strategy):
    """return vector
    >>> RPSTrainer(iterations=100).calculate_best_response_startegy(np.array([0.8, 0.1, 0.1]))
    array([0, 1, 0])
    >>> RPSTrainer(iterations=100).calculate_best_response_startegy(np.array([0.33, 0.33, 0.34]))
    array([1, 0, 0])
    >>> RPSTrainer(iterations=100).calculate_best_response_startegy(np.array([0, 1, 0]))
    array([0, 0, 1])
    """
    r = self.calculate_utility(np.array([1, 0, 0]), other_strategy)
    p = self.calculate_utility(np.array([0, 1, 0]), other_strategy)
    s = self.calculate_utility(np.array([0, 0, 1]), other_strategy)

    idx = np.argmax([r, p, s])
    strategy = np.array([0, 0, 0])
    strategy[idx] = 1.0

    return strategy


  def calculate_best_response_startegy_not_good(self, other_strategy):
    """return vector
    >>> RPSTrainer(iterations=100).calculate_best_response_startegy_not_good(np.array([0.8, 0.1, 0.1]))
    array([0, 1, 0])
    >>> RPSTrainer(iterations=100).calculate_best_response_startegy_not_good(np.array([0.33, 0.33, 0.34]))
    array([1, 0, 0])
    >>> RPSTrainer(iterations=100).calculate_best_response_startegy_not_good(np.array([0, 1, 0]))
    array([0, 0, 1])
    """
    max_hand_for_other = np.max(other_strategy)
    hand_idx = []
    for i, idx in enumerate(other_strategy):
      if idx == max_hand_for_other:
        hand_idx.append(i)

    strategy = np.array([0, 0, 0])

    if len(hand_idx) == 1:
      strategy[(hand_idx[0]+1)%3] = 1.0
    elif len(hand_idx) == 3:
      strategy[np.random.choice(hand_idx)%3] = 1.0
    else:
      if hand_idx == [0, 2]:
        strategy[0] = 1.0
      else:
        strategy[hand_idx[-1]%3] = 1.0

    return strategy


  #utilityを計算する
  def calculate_utility(self, avg_strategy_1, avg_strategy_2):
    r1 = avg_strategy_1[0]
    p1 =  avg_strategy_1[1]
    s1 =  avg_strategy_1[2]
    r2 =  avg_strategy_2[0]
    p2 =  avg_strategy_2[1]
    s2 =  avg_strategy_2[2]

    return r1*r2*0 + r1*p2*-1 + r1*s2*1 + p1*r2*1+p1*p2*0 + p1*s2*-1 + s1*r2*-1 + s1*p2*1 + s1*s2*0


  def train(self):
    utility_list = []
    for iteration in tqdm(range(self.iterations)):
      alpha = 1/(iteration+2)

      best_response_for_me = self.calculate_best_response_startegy(self.avg_opp_strategy)
      best_response_for_opp = self.calculate_best_response_startegy(self.avg_me_strategy)


      self.avg_me_strategy = (1-alpha)*self.avg_me_strategy + alpha*best_response_for_me
      self.avg_opp_strategy = (1-alpha)*self.avg_opp_strategy + alpha*best_response_for_opp

      utility_list.append(self.calculate_utility(self.avg_me_strategy, self.avg_opp_strategy))

    #描画
    plt.plot(range(len(utility_list)), utility_list)
    plt.show()


    return self.avg_me_strategy, self.avg_opp_strategy, self.calculate_utility(self.avg_me_strategy, self.avg_opp_strategy)


#学習
trainer0 = GameTrainer(iterations = 10000)
result0 = trainer0.train()
print("player0 strategy:", result0[0], "player1 strategy:", result0[1])

trainer1 = RPSTrainer(iterations = 1000)
result1  = trainer1.train()
print("player0 strategy:", result1[0], "player1 strategy:", result1[1])


doctest.testmod()
