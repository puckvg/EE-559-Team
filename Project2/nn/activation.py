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

    def backward(self):
        raise NotImplementedError

    def _grad_local(self, x, y):
        """
        Args:
            x: troch.tensor. Input tensor.
            y: torch.tensor. Target tensor.
        """
        pass
    
    
class ReLU(Activation):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """ Compute the activation.
        Args:
            x: torch.tensor. Input tensor.
        
        Returns:
            out: torch.tensor. Output tensor
        """
        out = f.relu(x)
        return out
    
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
        self.cache['dx_glob'] = dx_loc.T.mm(dy)
        
        return self.cache['dx_glob']
    
    def _grad_local(self, x):
        """ Compute local gradients of ReLU with respect to input and parameters. Store the gradients in the cache for the backward step.
        Args:
            x: torch.tensor. Input tensor.
        """
        
        self.cache['dx_loc'] = f.d_relu(x)
        


        
    