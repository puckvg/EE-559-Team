import torch

""" The functional.py contains the concrete implementations of specific. """

def mse(x, y, reduction='mean'):
    """ Compute the mean squared error.
        Args:
            x: torch.tensor. Input tensor.
            y: torch.tensor. Target tensor.

        Returns:
            ms_error: float. Mean squared error.
    """
    batch_size = x.shape[0]
    feature_dim = x.shape[1]
    
    # Samplewise MSE (vector)
    ms_error = (x - y).pow(2).sum(dim = 1) / feature_dim
    
    # Batchwise Average (scalar)
    ms_error = ms_error.sum(dim=0) / batch_size
    
    return ms_error


def d_mse(x, y, reduction='mean'):
    """ Compute the gradient of the mean squared error.
        Args:
            x: torch.tensor. Input tensor.
            y: torch.tensor. Target tensor.

        Returns:
            d_mse: float. Gradient of mean squared error.
    """
    d_mse = 2*(x - y) / (y.size(0) * y.size(1))
    return d_mse


def tanh(x):
    """ Compute tanh(x).
    Args:
        x: torch.tensor. Input tensor
    
    Returns:
        out: torch.tensor. Output tensor
    """
    out = x.tanh()
    return out


def d_tanh(x):
    """ Compute gradient of tanh(x)
    Args:
        x: torch.tensor. Input tensor
        
    Returns:
        out: torch.tensor. Output tensor
    """ 
    out = 1 - x.tanh().pow(2)
    return out


def relu(x):
    """ Compute ReLU(x)
    Args:
        x: torch.tensor. Input tensor
        
    Returns:
        out: torch.tensor. Output tensor
    """
    out = x.relu()
    return out


def d_relu(x):
    """ Compute gradient of ReLU(x)
    Args:
        x: torch.tensor. Input tensor
        
    Returns:
        out: torch.tensor. Output tensor
    """
    out = (x > 0) * 1
    return out