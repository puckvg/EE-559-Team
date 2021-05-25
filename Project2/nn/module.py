class Module(object):
    """Base Module from which nearly all nn modules inherit."""

    def __init__(self):
        """Initialize object of type Module with empty cache.
        The cache will be used to store information for subsequent passes such as the local gradient."""

        self.cache = {}

    def __call__(self, *args, **kwargs):
        """Built-in method to make instance of class callable.
        We use this to compute the forward pass and store the local gradients upon calling the module.

        Returns:
            torch.tensor: local gradients
        """

        out = self.forward(*args, **kwargs)
        self._grad_local(*args, **kwargs)
        return out

    def forward(self, *args, **kwargs):
        """Compute forward pass"""

    def backward(self, *args, **kwargs):
        """Compute backward pass"""

    def _grad_local(self, *args, **kwargs):
        """Compute local gradients of the Module with respect to input and parameters. Store the gradients in the cache for the backward step.
        Methods with an underscore at the beginning are pythons way of indicating that the methods is intended for class internal use."""
