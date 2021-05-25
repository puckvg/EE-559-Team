from nn.functional import *
from nn.module import Module


class Loss(Module):
    """The Loss Module is used to implement a node in the network that computes the loss.
    For the computation of any function the respective functional from functional.py should be used."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        """str: Return string representation"""

    def forward(self, x, y):
        """Compute the loss.
        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
        """

    def backward(self):
        """Backward pass.

        Returns:
            torch.tensor: Backpropagated gradient from the next layer.
        """

        return self.cache["dx_glob"]

    def _grad_local(self, x, y):
        """Compute local gradient with respect to the input x.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
        """


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def __str__(self):
        """str: Return string representation of MSELoss"""
        return "MSELoss()"

    def forward(self, x, y):
        """Compute the mean squared error.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.

        Returns:
            torch.tensor: Mean squared error.
        """

        return mse(x, y)

    def _grad_local(self, x, y):
        """Compute local gradient of MSE with respect to the input x.
        Store gradient in cache for backward step.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
        """

        self.cache["dx_glob"] = d_mse(x, y)
