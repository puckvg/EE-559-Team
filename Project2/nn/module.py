class Module(object):

    def __init__(self):
        """ Initialize object of type Module with empty cache.
        The cache will be used to store information for subsequent passes such as the local gradient. """
        self.cache = {}

    def __call__(self, *args, **kwargs):
        """ Built in method to make instance of class callable: https://www.geeksforgeeks.org/__call__-in-python/
        We use this to comppute the forward pass and store the local gradients upon calling the module. 
        Reference for *args, **kwargs: https://www.geeksforgeeks.org/args-kwargs-python/ """
        out = self.forward(*args, **kwargs)
        self._grad_local(*args, **kwargs)
        return out

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    def _grad_local(self, *args, **kwargs):
        """ Compute local gradients of the Module with respect to input and parameters. Store the gradients in the cache for the backward step.
        Methods with an underscore at the beginning are pythons way of indicating that the methods is intended for class internal use."""
        pass