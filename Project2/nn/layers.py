import torch
from nn.module import module

class Layer(module.Module):
    def update_param(self, *args, **kwargs):
        """ Update the params of the Layer based on the cached gradients """
        pass

    def _grad_local(self, x):
        pass

    def _get_param(self):
        """ Return the params of the Layer. """
        pass


class Linear(Layer):

    def __init__(self, dim_in, dim_out):
        """ Initialize object of type Linear with random parameters.
        Args:
            dim_in: int. Dimension of input.
            dim_out int. Dimension of output.

        Returns:
            output: Linear
        """
        raise NotImplementedError

    def forward(self, x):
        """ Calculate output.
        Args:
            x: torch.tensor. Input tensor.
        Returns:
            output: torch.tensor.
        """
        raise NotImplementedError

    def backward(self, dy):
        """ Compute gradients of input and parameters.
        Args:
            dy: torch.tensor: Backpropagated gradient from the next layer.
        Returns:
            output: torch.tensor: Gradient.
        """
        raise NotImplementedError

    def update_param(self, lr):
        """ Update parameter of the Linear Layer based on the cached gradients.
        Args:
            lr: float. Learning rate.
        """
        raise NotImplementedError

    def _get_param(self):
        """ Get parameters of the linear layer from the cache.
        Returns:
            w, b: torch.tensor.
        """
        raise NotImplementedError

    def _grad_local(self, x):
        """ Compute local gradients of Linear with respect to input and parameters. Store the gradients in the cache for the backward step.
        Args:
            x: torch.tensor. Input tensor.
        """
        raise NotImplementedError