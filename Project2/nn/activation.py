import torch
from nn.module import Module
import nn.functional as f

class Activation(Module):
    def forward(self, x):
        """ Compute the activation.
        Args:
            x: torch.tensor. Input tensor.
        """
        pass

    def backward(self, dy):
        """ Compute gradients of input.
        Args:
            dy: torch.tensor: Backpropagated gradient from the next layer.
            
        Returns:
            grad: torch.tensor. Gradient
        """
        
        # Read local gradients from cache
        dx_loc = self.cache['dx_loc']
        
        # Compute global gradients
        # self.cache['dx_glob'] = dx_loc.T.mv(dy)
        self.cache['dx_glob'] = dy.mul(dx_loc)
        
        return self.cache['dx_glob']

    def _grad_local(self, x, y):
        """
        Args:
            x: troch.tensor. Input tensor.
            y: torch.tensor. Target tensor.
        """
        pass

    
    
class ReLU(Activation):
    " ReLU activation function "
    
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU()"
        
    def forward(self, x):
        """ Compute the activation.
        Args:
            x: torch.tensor. Input tensor.
        
        Returns:
            out: torch.tensor. Output tensor.
        """
        out = f.relu(x)
        return out
    
    def _grad_local(self, x):
        """ Compute local gradients of ReLU with respect to input and parameters. Store the gradients in the cache for the backward step.
        Args:
            x: torch.tensor. Input tensor.
        """
        
        self.cache['dx_loc'] = f.d_relu(x)
        

class Tanh(Activation):
    " Tanh activation function "
    
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh"
        
    def forward(self, x):
        """ Compute the activation.
        Args:
            x: torch.tensor. Input tensor.
        
        Returns:
            out: torch.tensor. Output tensor.
        """
        out = f.tanh(x)
        return out
    
    def _grad_local(self, x):
        """ Compute local gradients of Tanh with respect to input and parameters. Store the gradients in the cache for the backward step.
        Args:
            x: torch.tensor. Input tensor.
        """
        self.cache['dx_loc'] = f.d_tanh(x)
    
