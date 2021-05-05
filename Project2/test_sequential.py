from unittest import TestCase
import torch, random
from nn.linear import Linear
from nn.loss import MSELoss
from nn.sequential import Sequential
from test_module import TestModule


n_tests = 10
max_dim = 5
max_n_layers = 100
thresh = 1e-3

class TestSequential(TestModule):
    def _forward_no_activation(self, in_dim, out_dim, n_layers):
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

        x, _ = self._gen_data(in_dim, out_dim)

        network_ours = Sequential(modules_ours, MSELoss())
        network_theirs = torch.nn.Sequential(*modules_theirs)

        out_ours = network_ours(x)
        out_theirs = network_theirs(x)

        assert out_ours.isclose(out_theirs, rtol=thresh).all(), 'Outputs must be equal'
        return network_ours, network_theirs

    def test_random_forward_no_activation(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            n_layers = random.randint(1, max_n_layers)
            self._forward_no_activation(in_dim, out_dim, n_layers)

    def _backward_no_activation(self, in_dim, out_dim, n_layers):
        network_ours, network_theirs = self._forward_no_activation(in_dim, out_dim, n_layers)
        x, y = self._gen_data(in_dim, out_dim)

        out_ours = network_ours(x)
        out_theirs = network_theirs(x)

        loss_ours = network_ours.loss(out_ours, y)
        loss_fn_theirs = torch.nn.MSELoss()
        loss_theirs = loss_fn_theirs(out_theirs, y)

        network_ours.backward()
        loss_theirs.backward()

        for m_ours, m_theirs in zip(network_ours.modules, list(network_theirs.children())):
            dw_ours = m_ours.cache['dw_glob']
            dw_theirs = m_theirs.weight.grad

            db_ours = m_ours.cache['db_glob']
            db_theirs = m_theirs.bias.grad

            assert db_ours.isclose(db_theirs, rtol=thresh).all(), 'Gradients of the bias must be equal'
            assert dw_ours.isclose(dw_theirs, rtol=thresh).all(), 'Gradients of the weights must be equal'

    def test_backward_single_layer(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._backward_no_activation(in_dim, out_dim, 1)

    def test_backward_n_layers(self):
        n = 3
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            self._backward_no_activation(in_dim, out_dim, n)

    def test_backward_random_layers(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            n_layers = random.randint(1, max_n_layers)
            self._backward_no_activation(in_dim, out_dim, n_layers)