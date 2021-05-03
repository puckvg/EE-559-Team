import torch 

class Optimizer():
    """
    base class for optimizers
    params : params to optimize as a dictionary (similar to cache) 
    lr : learning rate 
    """

    def __init__(self, params, lr=1e-3):
        self.lr = lr

        # check type 
        if not all(isinstance(k, torch.Tensor) for k in params): 
            raise TypeError('expected torch tensors as values in '
                            'params dict')

        # check names 
        required = ['w', 'b', 'dL_dw', 'dL_db']
        if not all(param in list(params.keys()) for param in required):
            raise NameError('expected params named w, b, dL_dw and dL_db')

        self.params = params

    def zero_grad(self):
        """sets gradients to zero"""
        self.params['dL_dw'].zero_()
        self.params['dL_db'].zero_()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer): 
    def __init__(self, params, lr=1e-3):
        super(Optimizer, self).__init__(params, lr=lr)

    def step(self): 
        self.params['w'] -= self.lr * self.params['dL_dw'] 
        self.params['b'] -= self.lr * self.params['dL_db'] 

