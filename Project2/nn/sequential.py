from nn.linear import Layer
from nn.module import Module


class Sequential(Module):
    """Sequential allows multiple layers to be combined in a network architecture."""

    def __init__(self, modules, loss_fn):
        """Create a sequential network.

        Args:
            modules (list(Module)): List of modules.
            loss_fn (str): loss function.

        Example:
            >>> LinNet = Sequential((
                    Linear(2, 25),
                    ReLU(),
                    Linear(25, 25),
                    ReLU(),
                    Linear(25, 25),
                    ReLU(),
                    Linear(25, 1)),
                    MSELoss()
                )
            >>> print(LinNet)
            Sequential(
                (0): Linear(in_features=2, out_features=25, bias=True)
                (1): ReLU()
                (2): Linear(in_features=25, out_features=25, bias=True)
                (3): ReLU()
                (4): Linear(in_features=25, out_features=25, bias=True)
                (5): ReLU()
                (6): Linear(in_features=25, out_features=1, bias=True)
            )
        """

        super().__init__()
        self.modules = modules
        self.loss_fn = loss_fn

    def __str__(self):
        """str: Return string representation of Sequential"""
        out = "Sequential(\n"
        for i, module in enumerate(self.modules):
            out += f"\t({i}): {str(module)}\n"
        out += ")"
        return out

    def print(self):
        """str: Print model architecture."""

        print("Sequential((")
        for module in self.modules:
            print(f"{module}, ")
        print("))")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Perform forward pass.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """

        out = x
        for module in self.modules:
            out = module(out)
        return out

    def backward(self):
        """Perform backward pass."""

        dy = self.loss_fn.backward()
        for module in reversed(self.modules):
            dy = module.backward(dy)

    def loss(self, x, y):
        """Compute loss between two tensors.

        Args:
            x (torch.tensor) : Input tensor
            y (torch.tensor) : Target tensor

        Returns:
            torch.tensor: Loss
        """

        return self.loss_fn(x, y)

    def update_params(self, optim, lr):
        """Update the parameters of the network iteratively
            according to the cached gradients at each module.

        Args:
            optim (string): The optimizer to use. options are 'adam' or 'sgd'
            lr (float): Learning rate
        """

        for module in self.modules:
            if isinstance(module, Layer):
                module._update_params(optim, lr)

    ### Methods for trainer ###
    def training_step(self, x, y):
        """Training step.

        Args:
            x (torch.tensor): Input tensor
            y (torch.tensor): Target tensor

        Returns:
            torch.tensor: Loss
        """

        out = self.forward(x)
        return self.loss(out, y)

    def validation_step(self, x, y):
        """Validation step.

        Args:
            x (torch.tensor): Input tensor
            y (torch.tensor): Target tensor

        Returns:
            torch.tensor: Loss
        """

        out = self.forward(x)
        return self.loss(out, y)

    def test_step(self, x, y):
        """Test step. Wrapper for validation_step.

        Args:
            x (torch.tensor): Input tensor
            y (torch.tensor): Target tensor

        Returns:
            torch.tensor: Loss
        """
        return self.validation_step(x, y)
