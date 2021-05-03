import torch
from nn.module import Module

class Layer(Module):
    def update_param(self, *args, **kwargs):
        """ Update the params of the Layer based on the cached gradients """
        pass

    def _grad_local(self, x):
        pass

    def param(self):
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
        super().__init__()
        
        # Initialize parameters
        self.cache['w'] = torch.empty((dim_out, dim_in)).normal_()
        self.cache['b'] = torch.empty((dim_out, )).normal_()

    def forward(self, x):
        """ Calculate output.
        Args:
            x: torch.tensor. Input tensor.
        Returns:
            output: torch.tensor.
        """
        w, b = self.param()
        return w.matmul(x) + b

    def backward(self, dy):
        """ Compute gradients of input and parameters.
        Args:
            dy: torch.tensor: Backpropagated gradient from the next layer.
        Returns:
            output: torch.tensor: Gradient.
        """

        # Read local gradients from cache
        dx_loc = self.cache['dx_loc']
        dw_loc = self.cache['dw_loc']

        # Compute global gradients
        self.cache['dx_glob'] = dx_loc.T.mm(dy)
        self.cache['dw_glob'] = dy.mm(dw_loc.T)
        self.cache['db_glob'] = dy
        return self.cache['dx_glob']

    def param(self):
        """ Get parameters of the linear layer from the cache.
        Returns:
            w, b: torch.tensor.
        """
        w = self.cache['w']
        b = self.cache['b']
        return w, b

    def _grad_local(self, x):
        """ Compute local gradients of Linear with respect to input and parameters. Store the gradients in the cache for the backward step.
        Args:
            x: torch.tensor. Input tensor.
        """
        w, _ = self.param()
        
        self.cache['dx_loc'] = w
        self.cache['dw_loc'] = x
        self.cache['db_loc'] = 1
    
    def _update_params(self, optim, lr):
        """ Update the parameters of this module according to the opimizer
            and the cached gradients """
        
        if optim == 'sgd':
            w, b = self.param()
            
            w -= lr * self.cache['dw_glob']
            b -= lr * self.cache['db_glob']

            self.cache['w'] = w
            self.cache['b'] = b