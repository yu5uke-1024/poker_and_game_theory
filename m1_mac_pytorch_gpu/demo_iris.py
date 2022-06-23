# -- Library --
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import time

time_start = time.time()

# アイリス　あやめの品種分類
class SL_Network(nn.Module):
    def __init__(self):
        super(SL_Network, self).__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        output = self.fc2(h1)

        return output


#device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
device = torch.device('cpu')

iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target.astype(np.int64)


X_train, X_test,  Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=3)

X_train, X_test, Y_train, Y_test = torch.from_numpy(X_train).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(Y_train).to(device), torch.from_numpy(Y_test).to(device)


model     = SL_Network().to(device)
optimizer = optim.SGD(model.parameters(),lr=0.05)
criterion = nn.CrossEntropyLoss()


# 学習回数
repeat = 1000

for epoch in tqdm(range(repeat)):

    output = model(X_train)


    loss = criterion(output, Y_train)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



model.eval()
with torch.no_grad():
    pred_model  = model(X_test)
    pred_result = torch.argmax(pred_model,1)

    accuracy = sum(pred_result.to('cpu').detach().numpy() == Y_test.to('cpu').detach().numpy()) / len(Y_test.to('cpu').detach().numpy())

time_end = time.time()

print("device:", device)
print("正答率:",accuracy)
print("経過時間:", time_end - time_start, "s")
