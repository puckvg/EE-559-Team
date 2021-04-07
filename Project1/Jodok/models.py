import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from torchvision.models import resnet18, googlenet
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
import torch

class BaseLightning(pl.LightningModule):
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


class BaseSimple(BaseLightning):
    def __init__(self, lr):
        super().__init__(lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss


class BaseCombined(BaseLightning):
    def __init__(self, lr):
        super().__init__(lr)

    def training_step(self, batch, batch_idx):
        x, y_class, y_target = batch
        d1, d2, out = self(x)
        loss_d1 = self.loss(d1, y_class[:, 0])
        loss_d2 = self.loss(d2, y_class[:, 1])
        loss_target = self.loss(out, y_target)
        loss = loss_d1 + loss_d2 + loss_target
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        _, _, out = self(x)
        loss = self.loss(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss


class Baseline(BaseSimple):
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 2)

    @auto_move_data
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


class LeNet(BaseSimple):
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    @auto_move_data
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


class ResNetMNIST(BaseSimple):
    def __init__(self, lr=0.001):
        super().__init__(lr)
        # define model and loss
        self.model = resnet18(num_classes=10, pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)

    @auto_move_data # this decorator automatically handles moving your tensors to GPU if required
    def forward(self, x):
        return self.model(x)


class CombinedNet(BaseCombined):
    def __init__(self, auxiliary, lr=0.001):
        super().__init__(lr)
        # define model and loss
        self.auxiliary = auxiliary
        self.loss = nn.CrossEntropyLoss()
        self.linear = nn.Linear(20, 2)

    @auto_move_data
    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        d1 = self.auxiliary(x1)
        #d1 = nn.functional.relu(d1)
        d2 = self.auxiliary(x2)
        #d2 = nn.functional.relu(d2)
        
        x = torch.cat((d1, d2), 1)
        x = self.linear(x)
        return d1, d2, x