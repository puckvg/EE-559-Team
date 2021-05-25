import nn.functional as f
from nn.module import Module


class Activation(Module):
    """Class to compute activation functions."""

    def __init__(self):
        super().__init__()

    def __str__(self):
        """str: Return string representation."""

    def forward(self, x):
        """Compute the activation.

        Args:
            x (torch.tensor): Input tensor.
        """

    def backward(self, dy):
        """Compute gradients of input.

        Args:
            dy (torch.tensor): Backpropagated gradient from the next layer.

        Returns:
            torch.tensor: Gradient
        """

        # Read local gradients from cache
        dx_loc = self.cache["dx_loc"]

        # Compute global gradients
        self.cache["dx_glob"] = dy.mul(dx_loc)

        return self.cache["dx_glob"]

    def _grad_local(self, x, y):
        """Compute local gradients of activation function.

        Args:
            x (torch.tensor): Input tensor.
            y (torch.tensor): Target tensor.
        """


class ReLU(Activation):
    "ReLU activation function class."

    def __init__(self):
        super().__init__()

    def __str__(self):
        """str: Return string representation of ReLU."""
        return "ReLU()"

    def forward(self, x):
        """Compute the ReLU.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Computed ReLU.
        """

        return f.relu(x)

    def _grad_local(self, x):
        """Compute local gradients of ReLU with respect to input and parameters. Store the gradients in the cache for the backward step.

        Args:
            x (torch.tensor): Input tensor.
        """

        self.cache["dx_loc"] = f.d_relu(x)


class Tanh(Activation):
    "Tanh activation function class."

    def __init__(self):
        super().__init__()

    def __str__(self):
        """str: Return string representation of Tanh."""
        return "Tanh()"

    def forward(self, x):
        """Compute the Tanh.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor: Computed Tanh.
        """

        return f.tanh(x)

    def _grad_local(self, x):
        """Compute local gradients of Tanh with respect to input and parameters. Store the gradients in the cache for the backward step.

        Args:
            x (torch.tensor): Input tensor.
        """

        self.cache["dx_loc"] = f.d_tanh(x)
