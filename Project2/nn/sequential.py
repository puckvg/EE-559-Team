import torch
from nn.module import Module
from nn.loss import Loss

class Sequential(Module):
    def __init__(self, modules, loss_fn):
        """ Create a new sequential network.
        Args:
            modules: list(Module). List of modules.
        """
        super().__init__()
        self.modules = modules
        self.loss_fn = loss_fn

    def __call__(self, x):
        out = self.forward(x)
        return out

    def forward(self, x):
        out = x
        for module in self.modules:
            out = module(out)
        return out

    def backward(self):
        dy = self.loss_fn.backward()
        for module in self.module.reverse():
            dy = module.backward(dy)
    
    def loss(self, x, y):
        loss = self.loss_fn(x, y)
        return loss

