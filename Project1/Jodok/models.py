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
    def __init__(self, lr, weight_aux):
        super().__init__(lr)
        self.weight_aux = weight_aux

    def training_step(self, batch, batch_idx):
        x, y_class, y_target = batch
        d1, d2, out = self(x)
        loss_d1 = self.loss(d1, y_class[:, 0])
        loss_d2 = self.loss(d2, y_class[:, 1])
        loss_target = self.loss(out, y_target)
        loss = self.weight_aux * (loss_d1 + loss_d2) + loss_target
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


class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, batch_normalization, skip_connections, lr=0.001):
        super().__init__()
        self.is_bn = batch_normalization
        self.is_skip = skip_connections

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)

        self.bn2 = nn.BatchNorm2d(nb_channels)

    @auto_move_data
    def forward(self, x):
        y = self.conv1(x)
        if self.is_bn: y = self.bn1(y)
        y = nn.functional.relu(y)
        y = self.conv2(y)
        if self.is_bn: y = self.bn2(y)
        if self.is_skip: y = y + x
        y = nn.functional.relu(y)

        return y

class ResNet(BaseSimple):
    def __init__(self, nb_channels, kernel_size, nb_blocks, lr=0.001):
        super().__init__(lr)
        self.conv1 = nn.Conv2d(1, nb_channels, kernel_size=1)
        self.resblocks = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size, True, True) for _ in range(nb_blocks))
        )
        self.avg = nn.AvgPool2d(kernel_size = 12)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(nb_channels, 10)

    @auto_move_data
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.resblocks(x)
        x = nn.functional.relu(self.avg(x))
        x = self.flat(x)
        x = self.fc(x)
        return x


class CombinedNet(BaseCombined):
    def __init__(self, auxiliary, lr=0.001, weight_aux=0.5):
        super().__init__(lr, weight_aux)
        # define model and loss
        self.auxiliary = auxiliary
        self.loss = nn.CrossEntropyLoss()
        self.linear = nn.Linear(20, 2)

    @auto_move_data
    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]

        d1 = self.auxiliary(x1)
        d2 = self.auxiliary(x2)
        
        x = torch.cat((d1, d2), 1)
        x = self.linear(x)
        return d1, d2, x

class FullyConv(BaseSimple):
    def __init__(self, lr=0.001):
        super().__init__(lr)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=(7, 7), stride=(7, 7))
        self.flat = nn.Flatten(start_dim=1)
    
    @auto_move_data
    def forward(self, x):
        x = torch.cat((x[:, 0:1], x[:, 1:2]), dim=2)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.flat(x)
        return x