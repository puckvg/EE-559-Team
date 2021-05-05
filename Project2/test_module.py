from unittest import TestCase
import torch, random
from nn.linear import Linear


n_tests = 10
max_dim = 5
max_n_layers = 100
thresh = 1e-1

class TestModule(TestCase):
    def _gen_data(self, in_dim, out_dim):
        x = torch.empty((in_dim)).normal_()
        y = torch.empty((out_dim)).normal_()
        return x, y

    def _init_modules(self, in_dim, out_dim):
        mod_ours = Linear(in_dim, out_dim)
        mod_theirs = torch.nn.Linear(in_dim, out_dim)
        mod_theirs.weight.data, mod_theirs.bias.data = mod_ours.param()
        return mod_ours, mod_theirs
    
    def _forward(self, in_dim, out_dim):
        raise NotImplementedError

    def _backward(self):
        raise NotImplementedError