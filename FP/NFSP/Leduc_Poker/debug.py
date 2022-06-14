import torch
print(torch.__version__)


a = torch.tensor([3])
device = torch.device("mps")
a.to(device)
print(a)


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(device)
