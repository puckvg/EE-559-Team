import torch 
from torch import nn 
from torch.nn import functional as F 
from torch import optim 


class BaseNet(nn.Module):
    def __init__(self, inp_channels, out_channels, n_hidden=50):
        super().__init__()
        self.conv1 = nn.Conv2d(inp_channels, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
       # self.dropout = nn.Dropout(p=0.25)
       # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, n_hidden)
        self.fc2 = nn.Linear(n_hidden, out_channels)

    def forward(self, x):
        # CONV LAYER 1
        x = F.relu(self.pool1(self.conv1(x)))

        # CONV LAYER 2
        x = F.relu(self.conv2(x))
      #  x = F.relu(self.pool2(x))

        # RESHAPE 
        x = x.view(x.size(0), -1)
        print('shape after reshaping', x.shape)

        # FC 1
        x = F.relu(self.fc1(x))

       # x = self.dropout(x)
        # FC2
        x = self.fc2(x)
        print('shape after fc2', x.shape)
        return x
