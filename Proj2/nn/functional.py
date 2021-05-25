"""functional.py contains the concrete implementations of specific functionals."""


def mse(x, y):
    """Compute the mean squared error.

    Args:
        x (torch.tensor): Input tensor.
        y (torch.tensor): Target tensor.

    Returns:
        torch.tensor: Mean squared error.
    """

    return (x - y).pow(2).sum(dim=1, keepdim=True).mean() / x.size(1)


def d_mse(x, y):
    """Compute the gradient of the mean squared error.

    Args:
        x (torch.tensor): Input tensor.
        y (torch.tensor): Target tensor.

    Returns:
        float: Gradient of mean squared error.
    """

    return 2 * (x - y) / x.size(0) / x.size(1)


def tanh(x):
    """Compute tanh(x).

    Args:
        x (torch.tensor): Input tensor

    Returns:
        torch.tensor: Output tensor
    """

    return x.tanh()


def d_tanh(x):
    """Compute gradient of tanh(x)

    Args:
        x (torch.tensor): Input tensor

    Returns:
        torch.tensor: Output tensor
    """

    return 1 - x.tanh().pow(2)


def relu(x):
    """Compute ReLU(x)

    Args:
        x (torch.tensor): Input tensor

    Returns:
        torch.tensor: Output tensor
    """

    return x.relu()


def d_relu(x):
    """Compute gradient of ReLU(x)

    Args:
        x (torch.tensor): Input tensor

    Returns:
        torch.tensor: Output tensor
    """

    return (x > 0) * 1
