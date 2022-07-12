# -- Library --

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

from tqdm import tqdm
import pandas as pd
import numpy as np
import time

time_start = time.time()

# -- config --
batch = 64
image_size = 28*28
#device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
device = torch.device('cpu')

learing_rate = 0.1
epochs = 100


transform = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=batch, shuffle=True
    )

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=batch, shuffle=True
    )




class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net(image_size, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learing_rate)


# -- train --

model.train()
for epoch in tqdm(range(epochs)):
    loss_sum = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()

        inputs = inputs.view(-1, image_size)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss_sum += loss

        loss.backward()
        optimizer.step()


model.eval()

loss_sum = 0
accuracy = 0
count = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)


        inputs = inputs.view(-1, image_size)
        outputs = model(inputs)


        loss_sum += criterion(outputs, labels)

        pred = outputs.argmax(1)
        count += len(pred)



        accuracy += sum(pred.to('cpu').detach().numpy() == labels.to('cpu').detach().numpy())

time_end = time.time()


print("正答率:", accuracy/count)
print("経過時間:", time_end - time_start, "s")
