from unittest import TestCase
import torch, random
from nn.linear import Linear
from nn.loss import MSELoss
from nn.sequential import Sequential
from test_module import TestModule


n_tests = 10
max_dim = 5
max_n_layers = 100
max_batch_size = 100
thresh = 1e-3

class TestLinear(TestModule):
    def _forward(self, batch_size, in_dim, out_dim):
        # Generate random test data
        x, y = self._gen_batch_data(batch_size, in_dim, out_dim)

        # Initialize modules
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)

        # Compute forward pass
        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        assert out_ours.isclose(out_theirs, rtol=thresh).all(), 'Outputs of linear must be the same'
        return mod_ours, mod_theirs

    def test_random_forward(self):
        for _ in range(n_tests):
            batch_size = random.randint(1, max_batch_size)
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._forward(batch_size, in_dim, out_dim)
    
    def _backward(self, batch_size, in_dim, out_dim):
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)
        
        loss_fn_ours = MSELoss()
        loss_fn_theirs = torch.nn.MSELoss()

        x, y = self._gen_batch_data(batch_size, in_dim, out_dim)

        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        loss_ours = loss_fn_ours(out_ours, y)
        loss_theirs = loss_fn_theirs(out_theirs, y)

        assert loss_ours.isclose(loss_theirs, rtol=thresh), 'Loss must be equal'
        
        dy = loss_fn_ours.backward()
        mod_ours.backward(dy)

        loss_fn_theirs = torch.nn.MSELoss()
        loss_theirs = loss_fn_theirs(out_theirs, y)
        loss_theirs.backward()

        if mod_ours.cache['dw_glob'].isclose(mod_theirs.weight.grad, rtol=thresh).all() == False:
            print(mod_ours.cache['dw_glob']) 
            print(mod_theirs.weight.grad)       
        
        assert mod_ours.cache['dw_glob'].isclose(mod_theirs.weight.grad, rtol=thresh).all(), 'Gradient of weights must be equal'
        assert mod_ours.cache['db_glob'].isclose(mod_theirs.bias.grad, rtol=thresh).all(), 'Gradient of bias must be equal'
    
    def test_small_backward(self):
        for _ in range(n_tests):
            in_dim = 3
            out_dim = 2
            batch_size = 2
            self._backward(batch_size, in_dim, out_dim)
    
    def test_random_backward(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            batch_size = random.randint(1, max_batch_size)
            self._backward(batch_size, in_dim, out_dim)