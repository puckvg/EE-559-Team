from unittest import TestCase
import torch, random
from nn.linear import Linear
from nn.loss import MSELoss
from nn.sequential import Sequential


n_tests = 10
max_dim = 100
max_n_layers = 10
thresh = 1e-3

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
    
    def _random_forward(self):
        raise NotImplementedError

    def _random_backward(self):
        raise NotImplementedError

class TestLayers(TestModule):
    def _random_forward(self):
        in_dim = random.randint(1, max_dim)
        out_dim = random.randint(1, max_dim)
        # Generate random test data
        x, _ = self._gen_data(in_dim, out_dim)

        # Initialize modules
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)

        # Compute forward pass
        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        assert (out_ours == out_theirs).all().item()

    def test_forward(self):
        for _ in range(n_tests):
            self._random_forward()


class TestLoss(TestModule):
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


class TestSequential(TestModule):
    def _random_forward_no_activation(self):
        n_layers = random.randint(1, max_n_layers)
        in_dim = random.randint(1, max_dim)
        hidden_in = in_dim
        hidden_out = random.randint(1, max_dim)
        
        modules_ours = []
        modules_theirs = []
        for _ in range(n_layers):
            m_ours, m_theirs = self._init_modules(hidden_in, hidden_out)
            modules_ours.append(m_ours)
            modules_theirs.append(m_theirs)
            hidden_in = hidden_out
            hidden_out = random.randint(1, max_dim)
        
        out_dim = hidden_out
        x, y = self._gen_data(in_dim, out_dim)

        network_ours = Sequential(modules_ours, MSELoss())
        network_theirs = torch.nn.Sequential(*modules_theirs)
        loss_fn_theirs = torch.nn.MSELoss()

        out_ours = network_ours(x)
        out_theirs = network_theirs(x)

        assert (out_ours - out_theirs).max().item() < thresh, 'Outputs must be equal'

#        loss_ours = network_ours.loss(out_ours, y)
#        loss_theirs = loss_fn_theirs(out_theirs, y)

    def test_forward_no_activation(self):
        for _ in range(n_tests):
            self._random_forward_no_activation()


