import torch
from nn.module import Module
from nn.functional import *


class Loss(Module):
    """ The Loss Module is used to implement a node in the network that computes the loss.
    For the computation of any function the respective functional from functional.py should be used. """

    def forward(self, x, y):
        """ Compute the loss.
        Args:
            x: torch.tensor. Input tensor.
            y: torch.tensor. Target tensor.
        """
        pass

    def backward(self):
        """ The backward method can be implemented in the generic Loss class
            as it should be the same for all Loss Modules. """
        return self.cache['dy']

    def _grad_local(self, x, y):
        """
        Args:
            x: troch.tensor. Input tensor.
            y: torch.tensor. Target tensor.
        """
        pass


class MSELoss(Loss):

    def forward(self, x, y):
        """ Compute the mean squared error.
        Args:
            x: torch.tensor. Input tensor.
            y: torch.tensor. Target tensor.

        Returns:
            output: float. Mean squared error.
        """
        return mse(x, y)

    def _grad_local(self, x, y):
        """ Compute local gradient of MSE with respect to the input x. 
        Store gradient in cache for backward step.
        Args:
            x: torch.tensor. Input tensor.
            y: torch.tensor. Target tensor.
        """
        grad = d_mse(x, y)
        self.cache['dy'] = grad
