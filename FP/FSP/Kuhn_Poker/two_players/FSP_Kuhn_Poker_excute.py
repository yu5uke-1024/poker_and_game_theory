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
import wandb

import FSP_Kuhn_Poker_trainer
import FSP_Kuhn_Poker_supervised_learning
import FSP_Kuhn_Poker_reinforcement_learning
import FSP_Kuhn_Poker_generate_data


#config
config = dict(
  iterations = 10**4,
  n= 2,
  m= 1,
  memory_size_rl= 30,
  memory_size_sl= 10000,
  rl_algo = ["epsilon-greedy", "boltzmann"][0],
  sl_algo = ["cnt", "mlp"][0],
  pseudo_code = ["general_FSP", "batch_FSP"][0],
  wandb_save = True
)



if config["wandb_save"]:
  wandb.init(project="FSP_project", name="kuhn_poker_{}_{}_{}".format(config["rl_algo"], config["sl_algo"], config["pseudo_code"]))
  wandb.config.update(config)


#train

kuhn_trainer = FSP_Kuhn_Poker_trainer.KuhnTrainer(
  train_iterations = config["iterations"]
  )


kuhn_trainer.train(
  n = config["n"],
  m = config["m"],
  memory_size_rl = config["memory_size_rl"],
  memory_size_sl = config["memory_size_sl"],
  rl_algo = config["rl_algo"],
  sl_algo = config["sl_algo"],
  pseudo_code = config["pseudo_code"],
  wandb_save = config["wandb_save"]
  )


#result

print("")
print("avg_utility", kuhn_trainer.eval_vanilla_CFR("", 0, 0, 1, 1))
print("final_exploitability", list(kuhn_trainer.exploitability_list.items())[-1])
print("")


result_dict_avg = {}
for key, value in sorted(kuhn_trainer.avg_strategy.items()):
  result_dict_avg[key] = value
df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
df = df.reindex(["J", "Jp", "Jb", "Jpb", "Q", "Qp", "Qb", "Qpb", "K", "Kp", "Kb", "Kpb"], axis="index")
df.index.name = "Node"

result_dict_br = {}
for key, value in sorted(kuhn_trainer.best_response_strategy.items()):
  result_dict_br[key] = value
df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=['Pass_br', "Bet_br"])
df1 = df1.reindex(["J", "Jp", "Jb", "Jpb", "Q", "Qp", "Qb", "Qpb", "K", "Kp", "Kb", "Kpb"], axis="index")
df1.index.name = "Node"

print(pd.concat([df, df1], axis=1))


doctest.testmod()
