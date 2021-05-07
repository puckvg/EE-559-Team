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

    def print(self):
        """ print model architecture"""
        print("Sequential((")
        for module in self.modules: 
            print(f"{module}, ")
        print("))")

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
        for module in reversed(self.modules):
            dy = module.backward(dy)
    
    def loss(self, x, y):
        loss = self.loss_fn(x, y)
        return loss

    def update_params(self, optim, lr):
        """ Update the parameters of the network iteratively
            according to the cached gradients at each module. 

        Args:
            optim: string. The optimizer to use.
                Example: 'adam', 'sgd'
            lr: float. Learning rate

        """
        for module in self.modules:
            module._update_params(optim, lr)


