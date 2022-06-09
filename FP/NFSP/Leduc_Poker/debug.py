import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
import time
import doctest
import copy
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from collections import defaultdict
from tqdm import tqdm
from collections import deque

print(datetime.datetime.now().month)

"""
NUM_ACTIONS = 2
NUM_PLAYERS = 2
STATE_BIT_LEN = (NUM_PLAYERS + 1) + 2*(NUM_PLAYERS *2 - 2)
hidden_units_num = 64


# _________________________________ SL NN class _________________________________
class SL_Network(nn.Module):
    def __init__(self, state_num, hidden_units_num):
        super(SL_Network, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num

        self.fc1 = nn.Linear(self.state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, 2)
        #self.fc3 = nn.Linear(self.state_num, 1)


        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        output = self.softmax(self.fc2(h1))

        #output = torch.sigmoid(self.fc3(x))


        return output


sl_network = SL_Network(state_num=STATE_BIT_LEN, hidden_units_num=hidden_units_num)

print(sl_network.state_dict())

model_path = 'model.pth'
sl_network.load_state_dict(torch.load(model_path))
a = torch.from_numpy(np.array([0, 1 ,0 ,0 ,1, 0, 0], dtype="float")).float().reshape(-1, STATE_BIT_LEN)
b = np.exp( np.array(sl_network.forward(a).reshape(-1,2).detach().numpy()[0]) )

print(b)
"""
