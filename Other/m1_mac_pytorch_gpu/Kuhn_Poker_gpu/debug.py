import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




# _________________________________ SL NN class _________________________________
class SL_Network(nn.Module):
    def __init__(self, state_num, hidden_units_num):
        super(SL_Network, self).__init__()
        self.state_num = state_num
        self.hidden_units_num = hidden_units_num

        self.fc1 = nn.Linear(self.state_num, self.hidden_units_num)
        self.fc2 = nn.Linear(self.hidden_units_num, 1)
        #self.fc3 = nn.Linear(self.state_num, 1)

        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.leaky_relu(self.fc1(x))

        #output = self.fc2(h1)
        h2 = self.dropout(h1)

        output = torch.sigmoid(self.fc2(h2))


        return output

print(torch.__version__)


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

a = torch.tensor([3, 1, 2])
a = a.to(device)


b = torch.tensor([3, 2, 2], device=device)

#print(a+b)


nn = SL_Network(state_num=7, hidden_units_num=32).to(device)

input = torch.rand(6,7).to(device)
output = nn.forward(input)
print(output)
