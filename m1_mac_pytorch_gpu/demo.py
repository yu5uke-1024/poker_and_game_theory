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


import warnings
warnings.filterwarnings('ignore')


class IrisModel(nn.Module):

    def __init__(self):
        super(IrisModel, self).__init__()

        # モジュールリストの中にモデル情報を記述
        self.model_info = nn.ModuleList([
             nn.Linear(4,6),   # (1)入力層：今回入力は4次元として設定・出力は6次元と任意に設定
             nn.Sigmoid(),     # (2)活性化関数(シグモイド)
             nn.Linear(6,3),   # (3)出力層：出力層のユニットは出力値分
            ])

    # 順方向の計算を記述
    def forward(self,x):
        for i in range(len(self.model_info)):
            x = self.model_info[i](x)
        return x


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target.astype(np.int64)

# 学習データ＆テストデータ分割
X_train, X_test,  Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=3)

# テンソル化
X_train, X_test, Y_train, Y_test = torch.from_numpy(X_train).to(device), torch.from_numpy(X_test).to(device), torch.from_numpy(Y_train).to(device), torch.from_numpy(Y_test).to(device)


model     = IrisModel().to(device)                # モデル
optimizer = optim.SGD(model.parameters(),lr=0.05) # パラメータ探索アルゴリズム(確率的勾配降下法 + 学習率lr=0.05を適用)
criterion = nn.CrossEntropyLoss()                 # 損失関数


# 学習回数
repeat = 1500

for epoch in tqdm(range(repeat)):
    ex_var = X_train  # 説明変数を作成
    target = Y_train  # 目的変数を作成

    # モデルのforward関数を用いた準伝播の予測(モデルの出力値算出)
    output = model(ex_var)

    # 上記出力値(output)と教師データ(target)を損失関数に渡し、損失関数を計算
    loss = criterion(output, target)

    # 勾配を初期化
    optimizer.zero_grad()

    # 損失関数の値から勾配を求め誤差逆伝播による学習実行
    loss.backward()

    # 学習結果に基づきパラメータを更新
    optimizer.step()

    print(output, target)


model.eval()
with torch.no_grad():
    pred_model  = model(X_test)              # テストデータでモデル推論
    pred_result = torch.argmax(pred_model,1) # 予測値

    # 正解率
    print(round(((Y_test == pred_result).sum()/len(pred_result)).item(),3))
