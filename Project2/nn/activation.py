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
    
    def backward(self):
        """ Compute gradients of input.
        Args:
            x: torch.tensor. Input tensor.
            
        Returns:
            grad: torch.tensor. Gradient
        """
        grad = f.d_relu(x)
        return grad
        
    