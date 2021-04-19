from torch import nn
import torch
from torch.nn.modules.dropout import Dropout

class AbstractModule(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        raise NotImplementedError()
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def accuracy(self, y_, y):
        return (y_.flatten() == y.flatten()).sum().item() / y_.flatten().size(0) * 100


class BaseModule(AbstractModule):
    """ All models should inherit from BaseModule to use the trainer """
    def __init__(self, lr):
        super().__init__(lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        return loss, acc
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        return loss, acc

class Siamese(BaseModule):
    """ Siamese modules can inherit from Siamese to use the trainer """
    def __init__(self, auxiliary, target=nn.Linear(20, 2), 
                 softmax=True, lr=0.001, weight_aux=0.5):
        """ 
        Args:
            auxiliary: Module. Network that produces the auxiliary loss.
            target: Module. Network that produces the target loss (starting from auxiliary layer)
                    for direct digit prediction + arithmetic comparison set target=None
            softmax: Boolean. Whether or not to use the softmax. 
            lr: float. Learning rate
            weight_aux: float. The weight for the auxiliary loss. weight_aux=1 means that it has the same weight as the target loss.
                        if weight_aux = 0, this is equivalent to just using the target loss 
        """
        super().__init__(lr)
        self.weight_aux = weight_aux
        self.auxiliary = auxiliary
        self.target = target
        self.softmax = softmax

    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        d1 = self.auxiliary(x1)
        d2 = self.auxiliary(x2)
                
        if self.target:
            x = torch.cat((d1, d2), 1)
            if self.softmax:
                x = nn.functional.softmax(x, dim=1)
            x = self.target(x)
        else: 
            p_d1 = torch.argmax(d1, dim=1)
            p_d2 = torch.argmax(d2, dim=1)
            x = (p_d1 <= p_d2).float()

        return d1, d2, x

    def training_step(self, batch, batch_idx):
        x, y_class, y_target = batch
        d1, d2, out = self(x)
        preds = torch.argmax(out, dim=1)
        loss_d1 = self.loss(d1, y_class[:, 0])
        loss_d2 = self.loss(d2, y_class[:, 1])

        if self.target: 
            loss_target = self.loss(out, y_target)
            loss = self.weight_aux * (loss_d1 + loss_d2) + loss_target
        else:
            loss = (loss_d1 + loss_d2) / 2 

        acc = self.accuracy(preds, y_target)
        return loss, acc
    
    def validation_step(self, batch, batch_idx):
        x, y_class, y_target = batch
        d1, d2, out = self(x)
        if self.target:
            loss = self.loss(out, y_target)
            preds = torch.argmax(out, dim=1)
        else: 
            loss_d1 = self.loss(d1, y_class[:, 0])
            loss_d2 = self.loss(d2, y_class[:, 1])
            loss = (loss_d1 + loss_d2) / 2
            preds = out

        acc = self.accuracy(preds, y_target)
        return loss, acc

class LeNet(BaseModule):
    """ LeNet-ish implementation of digit classifier """
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(x)

        x = self.flat(x)
        
        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)

        return x
    

class LinearBaseline(BaseModule):
    """ Simple linear model using nothing but target loss """
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(392, 150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 20)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 10)
        self.fc7 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = self.flat(x)
        
        x = self.fc1(x)
        x = nn.functional.relu(x)
        
        x = self.fc2(x)
        x = nn.functional.relu(x)
        
        for _ in range(1):
            x = self.fc3(x)
            x = nn.functional.relu(x)
        
        x = self.fc4(x)
        x = nn.functional.relu(x)
        
        for _ in range(2):
            x = self.fc5(x)
            x = nn.functional.relu(x)

        x = self.fc6(x)
        x = nn.functional.relu(x)
        
        x = self.fc7(x)
        
        return x 
    

class LinearAlpha(BaseModule):
    """ FC model with dropouts and batch normalization, going from pixels to aux output """
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.flat = nn.Flatten(start_dim=1)
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(196, 120)
        self.bn1 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(120, 50)
        self.bn2 = nn.BatchNorm1d(num_features=50)
        self.fc3 = nn.Linear(50, 50)
        self.bn3 = nn.BatchNorm1d(num_features=50)
        self.fc4 = nn.Linear(50, 20)
        self.bn4 = nn.BatchNorm1d(num_features=20)
        self.fc5 = nn.Linear(20, 10)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.flat(x)
        
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        for _ in range(2):
            x = self.fc3(x)
            x = self.bn3(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        
        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        # Layer 5
        x = self.fc5(x)
        return x
    
        
class LinearBeta(BaseModule):
    """ FC model with dropouts and batch normalization, going from aux output to target output """
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.fc1 = nn.Linear(20, 10)
        self.bn1 = nn.BatchNorm1d(num_features=10)
        self.fc2 = nn.Linear(10, 10)
        self.bn2 = nn.BatchNorm1d(num_features=10)
        self.fc3 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        # Second layer
        for _ in range(3):
            x = self.fc2(x)
            x = self.bn2(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        
        # Thrird layer
        x = self.fc3(x)
        
        return x
        

class TailLinear(BaseModule):
    def __init__(self, lr=0.001, label_encoded=False):
        """ 
        Example:
            Use label_encoded=True for class labels in [0, 9]
            Use label_encoded=False for one hot encoded labels
            Use label_encoded=False when network used as target network in siamese network
        """
        super().__init__(lr)
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.flat = nn.Flatten(start_dim=1)
        self.label_encoded = label_encoded

    def forward(self, x):
        if self.label_encoded:
            # one hot encode class labels
            x = nn.functional.one_hot(x).float()
            x = self.flat(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        preds = torch.argmax(out, dim=1)
        acc = self.accuracy(preds, y)
        return loss, acc
    
    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        preds = torch.argmax(out, dim=1)
        acc = self.accuracy(preds, y)
        return loss, acc

