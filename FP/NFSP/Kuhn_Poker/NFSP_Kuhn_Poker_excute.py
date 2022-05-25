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

import NFSP_Kuhn_Poker_trainer



#config
config = dict(
  iterations = 10**4,
  num_players = 2,
  memory_size_rl = 100,
  memory_size_sl = 100,
  wandb_save =  False
)



if config["wandb_save"]:
  wandb.init(project="Kuhn_Poker_{}players".format(config["num_players"]), name="nfsp")
  wandb.config.update(config)


#train

kuhn_trainer = NFSP_Kuhn_Poker_trainer.KuhnTrainer(
  train_iterations = config["iterations"],
  num_players= config["num_players"]
  )


kuhn_trainer.train(
  memory_size_rl = config["memory_size_rl"],
  memory_size_sl = config["memory_size_sl"],
  wandb_save = config["wandb_save"]
  )


#result

print("")
print("avg_utility", kuhn_trainer.eval_vanilla_CFR("", 0, 0, [1.0 for _ in range(config["num_players"])]))
print("final_exploitability", list(kuhn_trainer.exploitability_list.items())[-1])
print("")


result_dict_avg = {}
for key, value in sorted(kuhn_trainer.avg_strategy.items()):
  result_dict_avg[key] = value
df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
df.index.name = "Node"

result_dict_br = {}
for key, value in sorted(kuhn_trainer.best_response_strategy.items()):
  result_dict_br[key] = value
df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=['Pass_br', "Bet_br"])
df1.index.name = "Node"

#print(pd.concat([df, df1], axis=1))


doctest.testmod()