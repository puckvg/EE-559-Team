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
    
    def _forward(self, in_dim, out_dim):
        raise NotImplementedError

    def _backward(self):
        raise NotImplementedError

class TestLayers(TestModule):
    def _forward(self, in_dim, out_dim):
        # Generate random test data
        x, y = self._gen_data(in_dim, out_dim)
        self.x = x 
        self.y = y 

        # Initialize modules
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)

        # Compute forward pass
        out_ours = mod_ours(x)
        self.out_ours = out_ours

        x.requires_grad = True
        y.requires_grad = True 
        out_theirs = mod_theirs(x)
        self.out_theirs = out_theirs

        assert (out_ours == out_theirs).all().item()
        return mod_ours, mod_theirs

    def test_random_forward(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._forward(in_dim, out_dim)
            
    def _small_backward(self):
        in_dim = 3
        out_dim = 2

        mod_ours, mod_theirs = self._forward(in_dim, out_dim)
        
        loss_fn_ours = MSELoss()
        loss_ours = loss_fn_ours(self.out_ours, self.y)
        dy = loss_fn_ours.backward()
        mod_ours.backward(dy)

        loss_fn_theirs = torch.nn.MSELoss()
        y = self.y
        y.requires_grad = True 
        loss_theirs = loss_fn_theirs(self.out_theirs, y)
        loss_theirs.backward()

        assert (mod_ours.cache['dw_glob'] - mod_theirs.weight.grad).max().item() < thresh

    def test_backward(self): 
        for _ in range(n_tests):
            self._small_backward()


class TestLoss(TestModule):
    def _init_modules(self):
        loss_ours = MSELoss()
        loss_theirs = torch.nn.MSELoss()
        return loss_ours, loss_theirs

    def _forward(self, in_dim, out_dim):
        _, y = self._gen_data(in_dim, out_dim)
        _, y_ = self._gen_data(in_dim, out_dim)

        # Initialize losses
        loss_ours, loss_theirs = self._init_modules()

        # Compute forward pass
        out_ours = loss_ours(y_, y)

        y.requires_grad = True
        y_.requires_grad = True
        out_theirs = loss_theirs(y_, y)
        self.out_theirs = out_theirs

        assert (out_ours - out_theirs).item() < thresh
        return loss_ours, loss_theirs

    def test_forward(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._forward(in_dim, out_dim)

    def _backward(self, in_dim, out_dim):
        loss_ours, loss_theirs = self._forward(in_dim, out_dim)
        dy = loss_ours.backward()

        self.out_theirs.backward()

        # if we want these properties, we need to define a linear model 
        # otherwise we could just check the dy is the same ? 
        assert (loss_ours.cache['dw_glob'] == loss_theirs.weight.grad).all().item(), 'Gradient of weights must be equal'
        assert (loss_ours.cache['db_glob'] == loss_theirs.bias.grad).all().item(), 'Gradient of bias must be equal'

    def test_backward(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._backward(in_dim, out_dim)


class TestSequential(TestModule):
    def _random_forward_no_activation(self, in_dim, out_dim):
        n_layers = random.randint(1, max_n_layers)
        hidden_in = in_dim
        hidden_out = random.randint(1, max_dim)
        
        modules_ours = []
        modules_theirs = []
        for _ in range(n_layers - 1):
            m_ours, m_theirs = self._init_modules(hidden_in, hidden_out)
            modules_ours.append(m_ours)
            modules_theirs.append(m_theirs)
            hidden_in = hidden_out
            hidden_out = random.randint(1, max_dim)
        
        m_ours, m_theirs = self._init_modules(hidden_in, out_dim)
        modules_ours.append(m_ours)
        modules_theirs.append(m_theirs)

        x, y = self._gen_data(in_dim, out_dim)

        network_ours = Sequential(modules_ours, MSELoss())
        network_theirs = torch.nn.Sequential(*modules_theirs)

        out_ours = network_ours(x)
        out_theirs = network_theirs(x)

        assert (out_ours - out_theirs).max().item() < thresh, 'Outputs must be equal'
        return network_ours, network_theirs

    def test_forward_no_activation(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._random_forward_no_activation(in_dim, out_dim)

    def _random_backward_no_activation(self, in_dim, out_dim):
        network_ours, network_theirs = self._random_forward_no_activation(in_dim, out_dim)
        x, y = self._gen_data(in_dim, out_dim)

        out_ours = network_ours(x)
        out_theirs = network_theirs(x)

        loss_ours = network_ours.loss(out_ours, y)
        loss_fn_theirs = torch.nn.MSELoss()
        loss_theirs = loss_fn_theirs(out_theirs, y)

        network_ours.backward()
        loss_theirs.backward()
        for m_ours, m_theirs in zip(network_ours.modules, list(network_theirs.children())):
            assert (m_ours.cache['dw_glob'] - m_theirs.weight.grad).max() < thresh, 'Gradients of the weights must be equal'
            assert (m_ours.cache['db_glob'] - m_theirs.bias.grad).max() < thresh, 'Gradients of the bias must be equal'

    def test_backward_no_activation(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._random_backward_no_activation(in_dim, out_dim)
