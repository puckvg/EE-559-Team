from torch import nn
import torch, random
from torch.nn.modules.dropout import Dropout

class AbstractModule(nn.Module):
    """ Abstract Module that defines the necessary methods for the trainer. """
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
    def __init__(self, auxiliary, target=nn.Linear(20, 2), strategy='random',
                 softmax=True, argmax=True, lr=0.001, weight_aux=0.5):
        """ 
        Args:
            auxiliary: Module. Network that produces the auxiliary loss.
            target: Module. Network that produces the target loss (starting from auxiliary layer)
                    for direct digit prediction + arithmetic comparison set target=None
            softmax: Boolean. Whether or not to use the softmax. 
            argmax: Boolean. Whether or not to use the argmax.
            lr: float. Learning rate
            weight_aux: float. The weight for the auxiliary loss. weight_aux=1 means that it has the same weight as the target loss.
                        if weight_aux = 0, this is equivalent to just using the target loss
            strategy: string. The strategy to use on how to combine the different losses.
                        Possible strategies: [sum, random]
        """
        super().__init__(lr)
        self.weight_aux = weight_aux
        self.auxiliary = auxiliary
        self.target = target
        self.softmax = softmax
        self.argmax = argmax
        self.strategy = strategy

    def forward(self, x):
        # Split the input into the two images
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        # Run the input images through the auxiliary classifier
        d1 = self.auxiliary(x1)
        d2 = self.auxiliary(x2)

        if self.target:
            # Prepare input for target network: merge digit predictions
            x = torch.cat((d1, d2), 1)
            
            if self.softmax:
                # Prepare input for target network: merge the digit predictions
                # after applying softmax (to follow the probability constraint)
                x = torch.cat((nn.functional.softmax(d1, dim=1), nn.functional.softmax(d2, dim=1)), 1)
            
            if self.argmax:
                # Prepare input for target network: merge the digit predictions
                # after applying argmax (digit prediciton as labeled class)
                x = torch.cat((d1.argmax(dim=1).view(-1,1), d2.argmax(dim=1).view(-1,1)), 1)
                
            x = self.target(x)
        else:
            # Simulate the target network with the arithmetic operation '<='
            p_d1 = torch.argmax(d1, dim=1)
            p_d2 = torch.argmax(d2, dim=1)
            x = (p_d1 <= p_d2).float()

        return d1, d2, x

    def training_step(self, batch, batch_idx):
        x, y_class, y_target = batch
        
        # Compute loss_digit for both input images
        d1, d2, out = self(x)
        loss_d1 = self.loss(d1, y_class[:, 0])
        loss_d2 = self.loss(d2, y_class[:, 1])

        if self.target: 
            # Compute loss_target with the target network
            preds = torch.argmax(out, dim=1)
            loss_target = self.loss(out, y_target)

            # Equally weight loss_digit from both input images
            loss_digit = (loss_d1 + loss_d2) / 2
            
            if self.strategy == 'random':
                # Alternate the loss (loss_digit / loss_target) to optimize by choosing the loss at random
                decision = random.randint(0, 1)
                if decision:
                    loss = loss_target
                else:
                    loss = loss_digit
            elif self.strategy == 'sum':
                # Sum up the two losses (loss_digit / loss_target)
                loss = self.weight_aux * loss_digit + loss_target
            else:
                raise ValueError(f'Unknown strategy: {self.strategy}')
        else:
            # Simulate the target network with the arithmetic operation '<='
            preds = out 
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
    """ FC model going from pixels to aux output """
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(196, 160)
        self.fc2 = nn.Linear(160, 120)
        self.fc3 = nn.Linear(120, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 20)
        self.fc7 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.flat(x)
        
        # Layer 1
        x = self.fc1(x)
        x = nn.functional.relu(x)
        
        # Layer 2
        x = self.fc2(x)
        x = nn.functional.relu(x)
        
        # Layer 3
        x = self.fc3(x)
        x = nn.functional.relu(x)

        # Layer 4 
        x = self.fc4(x)
        x = nn.functional.relu(x)

        # Layer 5
        for _ in range(3):
            x = self.fc5(x)
            x = nn.functional.relu(x)
        
        # Layer 6
        x = self.fc6(x)
        x = nn.functional.relu(x)
        
        # Layer 7
        x = self.fc7(x)
        return x
    
        
class LinearBeta(BaseModule):
    """ FC model going from aux output to target output """
    def __init__(self, lr=0.001, label_encoded=True):
        super().__init__(lr)
        self.fc1 = nn.Linear(20, 10)
        self.bn1 = nn.BatchNorm1d(num_features=10)
        self.fc2 = nn.Linear(10, 10)
        self.bn2 = nn.BatchNorm1d(num_features=10)
        self.fc3 = nn.Linear(10, 2)
        self.flat = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.label_encoded = label_encoded
        
    def forward(self, x):
        if self.label_encoded:
            # one hot encode class labels
            x = nn.functional.one_hot(x, num_classes=10).float()
            x = self.flat(x)
        
        # First layer
        x = self.fc1(x)
        # x = self.bn1(x)
        x = nn.functional.relu(x)
        # x = self.dropout(x)
        
        # Second layer
        for _ in range(3):
            x = self.fc2(x)
            # x = self.bn2(x)
            x = nn.functional.relu(x)
            # x = self.dropout(x)
        
        # Third layer
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
        

class TailLinear(BaseModule):
    def __init__(self, lr=0.001, label_encoded=True):
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
            x = nn.functional.one_hot(x, num_classes=10).float()
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


class SequencePretrained(BaseModule):
    def __init__(self, auxiliary, target, softmax=True, lr=0.001):
        super().__init__(lr)
        self.auxiliary = auxiliary
        self.target = target
        self.softmax = softmax
        
    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        d1 = self.auxiliary(x1)
        d2 = self.auxiliary(x2)
        d1 = d1.argmax(dim=1).view(-1, 1)
        d2 = d2.argmax(dim=1).view(-1, 1)
                
        x = torch.cat((d1, d2), 1)
        x = self.target(x)
        
        return x

    def validation_step(self, batch, batch_idx):
        x, _, y_target = batch
        out = self(x)

        loss = self.loss(out, y_target)
        preds = torch.argmax(out, dim=1)

        acc = self.accuracy(preds, y_target)
        return loss, acc

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    

