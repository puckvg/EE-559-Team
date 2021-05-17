from unittest import TestCase
from test_module import TestModule
import torch 
from nn.loss import MSELoss
from nn.sequential import Sequential
from nn.activation import ReLU, Tanh

n_tests = 10 
max_dim = 5 
max_n_layers = 100
thresh = 1e-3 

class TestSGDModule(TestModule):
    def _step(self, batch_size, in_dim, out_dim, lr):
        x, y = self._gen_batch_data(batch_size, in_dim, out_dim)

        # Initialize modules 
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)

        # Initialize optimizers 
        opt_ours = 'sgd'
        opt_theirs = torch.optim.SGD(mod_theirs.parameters(), lr=lr,
                                    momentum=0)

        # Initialize loss 
        loss_fn_ours = MSELoss()
        loss_fn_theirs = torch.nn.MSELoss()

        mod_theirs.zero_grad()

        # compute forward pass
        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        # loss 
        loss_ours = loss_fn_ours(out_ours, y)
        loss_theirs = loss_fn_theirs(out_theirs, y)

        # backward
        dy = loss_fn_ours.backward()
        mod_ours.backward(dy)

        loss_theirs.backward()

        # opt 
        opt_theirs.step()
        mod_ours._update_params(optim=opt_ours, lr=lr)

       # set nans to the same number for isclose test 
        mod_ours.cache['w'][torch.isnan(mod_ours.cache['w'])] = 0.
        mod_ours.cache['b'][torch.isnan(mod_ours.cache['b'])] = 0.

        with torch.no_grad():
            mod_theirs.weight[torch.isnan(mod_theirs.weight)] = 0.
            mod_theirs.bias[torch.isnan(mod_theirs.bias)] = 0.

        assert mod_ours.cache['w'].isclose(mod_theirs.weight.T, rtol=thresh).all(), 'weights after SGD step must be the same'
        assert mod_ours.cache['b'].isclose(mod_theirs.bias, rtol=thresh).all(), 'bias after SGD step must be the same'

    def test_optim_step(self):
        for _ in range(n_tests):
            in_dim = 3
            out_dim = 2
            batch_size = 4
            lr = 0.01 
            self._step(batch_size, in_dim, out_dim, lr)

class TestAdamModule(TestModule):
    def _step(self, batch_size, in_dim, out_dim, lr):
        x, y = self._gen_batch_data(batch_size, in_dim, out_dim)

        # Initialize modules 
        mod_ours, mod_theirs = self._init_modules(in_dim, out_dim)

        # Initialize optimizers 
        opt_ours = 'adam'
        opt_theirs = torch.optim.Adam(mod_theirs.parameters(), lr=lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-8)

        # Initialize loss 
        loss_fn_ours = MSELoss()
        loss_fn_theirs = torch.nn.MSELoss()

        mod_theirs.zero_grad()

        # compute forward pass
        out_ours = mod_ours(x)
        out_theirs = mod_theirs(x)

        # loss 
        loss_ours = loss_fn_ours(out_ours, y)
        loss_theirs = loss_fn_theirs(out_theirs, y)

        # backward
        dy = loss_fn_ours.backward()
        mod_ours.backward(dy)

        loss_theirs.backward()

        # opt 
        opt_theirs.step()
        mod_ours._update_params(optim=opt_ours, lr=lr)

        # set nans to the same number for isclose test 
        mod_ours.cache['w'][torch.isnan(mod_ours.cache['w'])] = 0.
        mod_ours.cache['b'][torch.isnan(mod_ours.cache['b'])] = 0.
        with torch.no_grad():
            mod_theirs.weight[torch.isnan(mod_theirs.weight)] = 0.
            mod_theirs.bias[torch.isnan(mod_theirs.bias)] = 0.

        assert mod_ours.cache['w'].isclose(mod_theirs.weight.T, rtol=thresh).all(), 'weights after Adam step must be the same'
        assert mod_ours.cache['b'].isclose(mod_theirs.bias, rtol=thresh).all(), 'bias after Adam step must be the same'

    def test_optim_step(self):
        for _ in range(n_tests):
            in_dim = 3
            out_dim = 2
            lr = 0.01 
            batch_size = 16
            self._step(batch_size, in_dim, out_dim, lr)

