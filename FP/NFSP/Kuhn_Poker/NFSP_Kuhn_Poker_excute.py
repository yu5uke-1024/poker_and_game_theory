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
from collections import deque
import wandb

import NFSP_Kuhn_Poker_trainer



#config
config = dict(
  iterations = 10**5,
  num_players = 2,
  eta = 0.1,
  memory_size_rl = 10**3,
  memory_size_sl = 10**3,
  wandb_save = True
)



if config["wandb_save"]:
  wandb.init(project="Kuhn_Poker_{}players".format(config["num_players"]), name="NFSP")
  wandb.config.update(config)
  wandb.define_metric("exploitability", summary="last")
  wandb.define_metric("avg_utility", summary="last")


#train

kuhn_trainer = NFSP_Kuhn_Poker_trainer.KuhnTrainer(
  train_iterations = config["iterations"],
  num_players= config["num_players"]
  )


kuhn_trainer.train(
  eta = config["eta"],
  memory_size_rl = config["memory_size_rl"],
  memory_size_sl = config["memory_size_sl"],
  wandb_save = config["wandb_save"],
  )



#result
if not config["wandb_save"]:
  print("avg_utility", list(kuhn_trainer.avg_utility_list.items())[-1])
  print("final_exploitability", list(kuhn_trainer.exploitability_list.items())[-1])


result_dict_avg = {}
for key, value in sorted(kuhn_trainer.avg_strategy.items()):
  result_dict_avg[key] = value
df = pd.DataFrame(result_dict_avg.values(), index=result_dict_avg.keys(), columns=['Pass_avg', "Bet_avg"])
df.index.name = "Node"

result_dict_br = {}
for key, value in sorted(kuhn_trainer.epsilon_greedy_q_learning_strategy.items()):
  result_dict_br[key] = value
df1 = pd.DataFrame(result_dict_br.values(), index=result_dict_br.keys(), columns=['Pass_br', "Bet_br"])
df1.index.name = "Node"

df2 = pd.concat([df, df1], axis=1)


if config["wandb_save"]:
  tbl = wandb.Table(data=df2)
  tbl.add_column("Node", [i for i in df2.index])
  wandb.log({"table:":tbl})
  wandb.save()
else:
  print(df2)



doctest.testmod()
