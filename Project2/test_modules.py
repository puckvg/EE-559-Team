from unittest import TestCase
import torch, random
from nn.linear import Linear
from nn.loss import MSELoss


n_tests = 10
max_dim = 100
thresh = 1e-3

class TestLayers(TestCase):
    def _gen_data(self):
        in_dim = random.randint(1, max_dim)
        out_dim = random.randint(1, max_dim)
        x = torch.empty((in_dim)).normal_()
        y = torch.empty((in_dim)).normal_()
        return x, y, in_dim, out_dim

    def _init_modules(self, in_dim, out_dim):
        mod_ours = Linear(in_dim, out_dim)
        mod_theirs = torch.nn.Linear(in_dim, out_dim)
        return mod_ours, mod_theirs

    def _random_forward(self):
        # Generate random test data
        x, _, in_dim, out_dim = self._gen_data()

        # Initialize modules
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)
        
        # Set params to be equal
        mod_theirs.weight.data, mod_theirs.bias.data = mod_ours.param()

        # Compute forward pass
        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        assert (out_ours == out_theirs).all().item()

    def test_forward(self):
        for _ in range(n_tests):
            self._random_forward()


class TestLoss(TestCase):
    def _gen_data(self):
        dim = random.randint(1, max_dim)
        y_, y = torch.empty((dim)).normal_(), torch.empty((dim)).normal_()
        return y_, y

    def _init_modules(self):
        loss_ours = MSELoss()
        loss_theirs = torch.nn.MSELoss()
        return loss_ours, loss_theirs

    def _random_forward(self):
        # Generate random test data
        y_, y = self._gen_data()

        # Initialize losses
        loss_ours, loss_theirs = self._init_modules()

        # Compute forward pass
        out_ours = loss_ours(y_, y)
        out_theirs = loss_theirs(y_, y)

        assert (out_ours - out_theirs).item() < thresh

    def test_forward(self):
        for _ in range(n_tests):
            self._random_forward()

    def _random_backward(self):
        y_, y = self._gen_data()
        loss_ours, loss_theirs = self._init_modules()

        _ = loss_ours(y_, y)
        _ = loss_theirs(y_, y)

        loss_ours.backward()
        loss_theirs.backward()
        
        assert (loss_ours.cache['dw_glob'] == loss_theirs.weight.grad).all().item(), 'Gradient of weights must be equal'
        assert (loss_ours.cache['db_glob'] == loss_theirs.bias.grad).all().item(), 'Gradient of bias must be equal'

