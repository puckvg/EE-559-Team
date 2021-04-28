import torch
from nn.module import Module
from nn.functional import *

class Activation(Module):
    def forward(self, x, y):
        """ Compute the activation.
        Args:
            x: torch.tensor. Input tensor.
            y: torch.tensor. Target tensor.
        """
        pass

    def backward(self):
        raise NotImplementedError

    def _grad_local(self, x, y):
        """
        Args:
            x: troch.tensor. Input tensor.
            y: torch.tensor. Target tensor.
        """
        pass