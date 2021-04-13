# Imports
import torch

from torch import nn
from torch.nn import functional as F


class CombinedBaseModel(nn.Module):
    def __init__(self, ConvNet, AuxNet, ClassNet, mode='train_no_aux'):
        super().__init__()
        
        self.ConvNet = ConvNet
        self.AuxNet = AuxNet
        self.ClassNet = ClassNet
        
        self.loss_mode = mode
        
    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]
        
        d1 = self.ConvNet(x1)
        d2 = self.ConvNet(x2)
        
        if self.loss_mode == 'train_without_aux':
            x = torch.cat((d1, d2), dim=1)
            x = self.ClassNet(x)
            return x
        elif self.loss_mode == 'train_aux_only':
            x1 = self.AuxNet(d1)
            x2 = self.AuxNet(d2)
            x = torch.cat((x1, x2), dim=0)
            return x
        elif self.loss_mode == 'train_with_aux':
            x1 = self.AuxNet(d1)
            x2 = self.AuxNet(d2)
            x = torch.cat((d1, d2), 1)
            x = self.ClassNet(x)
            x_aux = torch.cat((x1, x2), dim=0)
            return x, x_aux
        else:
            raise NotImplementedError("Train mode not found")


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define layers
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=16,
                              kernel_size = 5,
                              padding = 2)
        self.conv_mid = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size = 3,
                               padding = 2)
        self.conv_out = nn.Conv2d(in_channels=16, out_channels= 16,
                               kernel_size = 3,
                               padding = 2)
        self.bn = nn.BatchNorm2d(num_features=16)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        # First layer
        x = self.conv_in(x)
        x = self.bn(x)
        # [1000, 16, 14, 14]
        x = F.relu(x)
        x = self.pool(x)
        # [1000, 16, 7, 7]
        
        # Second layer
        x = self.conv_mid(x)
        x = self.bn(x)
        x = F.relu(x)
        # [1000, 16, 9, 9]
        
        # Third layer
        x = self.conv_mid(x)
        x = self.bn(x)
        x = F.relu(x)
        # [1000, 16, 11, 11]
        
        # Fourth layer
        x = self.conv_out(x)
        x = self.bn(x)
        x = F.relu(x)
        # [1000, 16, 13, 13]
        x = self.pool(x)
        # [1000, 16, 6, 6]
        
        x = x.view(x.size(0), -1)
        # [1000, 576]
        
        return x
    
    
class AuxNet(nn.Module):
    def __init__(self, in_features, out_features = 10):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
 
        
class ClassNet(nn.Module):
    def __init__(self, in_features, out_features = 2):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
        
class FelixNet(CombinedBaseModel):
    def __init__(self):
        super().__init__(ConvNet = SimpleConvNet(), 
                         AuxNet = AuxNet(), 
                         ClassNet = ClassNet())