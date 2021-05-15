import torch
from nn.module import Module
from nn.functional import *


class Loss(Module):
    """The Loss Module is used to implement a node in the network that computes the loss.
    For the computation of any function the respective functional from functional.py should be used."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        """Return string representation"""
        pass 

    def forward(self, x, y):
        """ Compute the loss.
        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
        """
        pass

    def backward(self):
        """Backward pass. 

        Returns: 
            dy (torch.tensor): Backpropagated gradient from the next layer.
        """
        dy = self.cache['dy']
        return dy

    def _grad_local(self, x, y):
        """Compute local gradient with respect to the input x.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
        """
        pass



class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "MSELoss()"

    def forward(self, x, y):
        """Compute the mean squared error.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.

        Returns:
            output (torch.tensor): Mean squared error.
        """
        output = mse(x, y)
        return output

    def _grad_local(self, x, y):
        """Compute local gradient of MSE with respect to the input x. 
        Store gradient in cache for backward step.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
        """
        grad = d_mse(x, y)
        self.cache['dy'] = grad
