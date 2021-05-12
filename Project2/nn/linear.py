import torch, math
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
        self.cache['w'] = torch.empty((dim_out, dim_in))
        self.cache['b'] = torch.empty((dim_out, ))
        self._init_param()
        
        # Initialize optimizer params (only needed for ADAM optimizer)
        # momentum
        self.cache['m_dw'] = 0.
        self.cache['m_db'] = 0.
        # rms 
        self.cache['v_dw'] = 0.
        self.cache['v_db'] = 0.
        # timestep 
        self.cache['t'] = 1

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

    def _init_param(self):
        stdv = 1. / math.sqrt(self.cache['w'].size(1))
        self.cache['w'].uniform_(-stdv, stdv)
        self.cache['b'].uniform_(-stdv, stdv)
    
    def _update_params(self, optim, lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        """ Update the parameters of this module according to the opimizer
            and the cached gradients 
            Args:
            lr : step size (for all optimizers)
            beta_1 : first order exp decay (for adam)
            beta_2 : second order exp decay (for adam)
            epsilon : small param to prevent zero division (for adam)"""
        
        if optim == 'sgd':
            w, b = self.param()
            
            w -= lr * self.cache['dw_glob']
            b -= lr * self.cache['db_glob']

            self.cache['w'] = w
            self.cache['b'] = b
            
        elif optim == 'adam':
            w, b = self.param()
            t = self.cache['t']

            # momentum 
            m_dw = beta_1 * self.cache['m_dw'] + (1 - beta_1) * self.cache['dw_glob'] 
            m_db = beta_1 * self.cache['m_db'] + (1 - beta_1) * self.cache['db_glob']

            # rms 
            v_dw = beta_2 * self.cache['v_dw'] + (1 - beta_2) * self.cache['dw_glob']**2
            v_db = beta_2 * self.cache['v_db'] + (1 - beta_2) * self.cache['db_glob']

            # bias correction 
            m_dw_corr = m_dw / (1 - beta_1**t)
            m_db_corr = m_db / (1 - beta_1**t)
            v_dw_corr = v_dw / (1 - beta_2**t)
            v_db_corr = v_db / (1 - beta_2**t)

            w -= lr * (m_dw_corr / (torch.sqrt(v_dw_corr) + epsilon))
            b -= lr * (m_db_corr / (torch.sqrt(v_db_corr) + epsilon))

            self.cache['m_dw'] = m_dw
            self.cache['m_db'] = m_db
            self.cache['v_dw'] = v_dw 
            self.cache['v_db'] = v_db
            self.cache['t'] += 1
            self.cache['w'] = w
            self.cache['b'] = b
            
        else:
            raise NotImplementedError
