import torch

""" The functional.py contains the concrete implementations of specific functionals. """

def mse(x, y):
    """ Compute the mean squared error.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.

        Returns:
            ms_error (torch.tensor): Mean squared error.
    """
    
    ms_error = (x-y).pow(2).sum(dim=1, keepdim=True).mean() / x.size(1)

    return ms_error


def d_mse(x, y):
    """ Compute the gradient of the mean squared error.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.

        Returns:
            d_mse (float): Gradient of mean squared error.
    """
    
    d_mse = 2*(x - y) / x.size(0) / x.size(1)
    return d_mse


def tanh(x):
    """ Compute tanh(x).

    Args:
        x (torch.tensor): Input tensor
    
    Returns:
        out (torch.tensor): Output tensor
    """

    out = x.tanh()
    return out


def d_tanh(x):
    """ Compute gradient of tanh(x)

    Args:
        x (torch.tensor): Input tensor
        
    Returns:
        out (torch.tensor): Output tensor
    """ 

    out = 1 - x.tanh().pow(2)
    return out


def relu(x):
    """ Compute ReLU(x)

    Args:
        x (torch.tensor): Input tensor
        
    Returns:
        out (torch.tensor): Output tensor
    """

    out = x.relu()
    return out


def d_relu(x):
    """ Compute gradient of ReLU(x)

    Args:
        x (torch.tensor): Input tensor
        
    Returns:
        out (torch.tensor): Output tensor
    """
    
    out = (x > 0) * 1
    return out
