from unittest import TestCase
import torch, random
from nn.linear import Linear
from nn.loss import MSELoss
from nn.sequential import Sequential
from test_module import TestModule


n_tests = 10
max_dim = 5
max_n_layers = 100
thresh = 1e-1

class TestLinear(TestModule):
    def _forward(self, in_dim, out_dim):
        # Generate random test data
        x, y = self._gen_data(in_dim, out_dim)

        # Initialize modules
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)

        # Compute forward pass
        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        assert (out_ours == out_theirs).all().item()
        return mod_ours, mod_theirs

    def test_random_forward(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._forward(in_dim, out_dim)
            
    def _backward(self, in_dim, out_dim):
        mod_ours, mod_theirs = self._forward(in_dim, out_dim)
        
        loss_fn_ours = MSELoss()
        loss_fn_theirs = torch.nn.MSELoss()

        x, y = self._gen_data(in_dim, out_dim)

        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        loss_ours = loss_fn_ours(out_ours, y)
        loss_theirs = loss_fn_theirs(out_theirs, y)

        dy = loss_fn_ours.backward()
        mod_ours.backward(dy)

        loss_fn_theirs = torch.nn.MSELoss()
        loss_theirs = loss_fn_theirs(out_theirs, y)
        loss_theirs.backward()

        assert (mod_ours.cache['dw_glob'] - mod_theirs.weight.grad).max().item() < thresh
    
    def test_small_backward(self):
        for _ in range(n_tests):
            in_dim = 3
            out_dim = 2
            self._backward(in_dim, out_dim)
    
    def test_random_backward(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._backward(in_dim, out_dim)