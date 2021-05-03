from unittest import TestCase
import torch, random
from nn.linear import Linear


n_tests = 10
max_dim = 100

class TestLayers(TestCase):
    def _random_forward(self):
        # Generate random test data
        in_dim, out_dim = random.randint(1, max_dim), random.randint(1, max_dim)
        x = torch.empty((in_dim,)).normal_()

        # Initialize modules
        mod_ours = Linear(in_dim, out_dim)
        mod_theirs = torch.nn.Linear(in_dim, out_dim)
        
        # Set params to be equal
        mod_theirs.weight.data, mod_theirs.bias.data = mod_ours.param()

        # Compute forward pass
        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        assert (out_ours == out_theirs).all().item()

    def test_forward(self):
        for _ in range(n_tests):
            self._random_forward()

    #def test_backward(self):
        # need loss module for that
