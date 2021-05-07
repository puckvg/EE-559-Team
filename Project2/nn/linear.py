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
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        # Initialize parameters
        self.cache['w'] = torch.empty((dim_out, dim_in)).normal_()
        self.cache['b'] = torch.empty((dim_out, )).normal_()
        
        # Initialize optimizer params (only needed for ADAM optimizer)
        self.cache['t_adam'] = 0
        self.cache['m_adam'] = 0
        self.cache['v_adam'] = 0

    def __str__(self):
        return f"Linear({self.dim_in}, {self.dim_out})"

    def forward(self, x):
        """ Calculate output.
        Args:
            x: torch.tensor. Input tensor.
        Returns:
            output: torch.tensor.
        """
        w, b = self.param()
        return w.mv(x) + b

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
        self.cache['dx_glob'] = dx_loc.T.mv(dy)
        self.cache['dw_glob'] = dy.view(-1, 1).matmul(dw_loc.view(1, -1))
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
            
        if optim == 'adam':
            # TODO:
            # - code below only for one-dimensional gradients, it still has to be adapted to multiple dimensions
            
            # Hypterparameters
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 1e-08
            
            # Get iteration number from cache
            t = self.cache['t_adam']
            
            # Get moments from cache
            m = self.cache['m_adam']
            v = self.cache['v_adam']
            
            # Get gradients and parameters
            dw_glob, db_glob = self.cache['dw_glob'], self.cache['db_glob']
            w, b = self.param()
            
            # Assemble gradients and parameters (merge bias and weights)
            # TODO
            g = [] # dw_glob + db_glob
            w = [] # w + b
            
            # Update moments
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * g.pow(2)
            
            m_hat = m / (1 - beta_1.pow(t))
            v_hat = v / (1 - beta_2.pow(t))
            
            # Update parameters
            w = w - lr * m_hat / (v_hat.sqrt() + epsilon)
            
            # Split parameters into weight  and bias
            # TODO
        
