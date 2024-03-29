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
max_batch_size = 100

class TestLoss(TestModule):
    def _init_modules(self):
        loss_ours = MSELoss()
        loss_theirs = torch.nn.MSELoss()
        return loss_ours, loss_theirs

    def _forward(self, batch_size, in_dim, out_dim):
        _, y = self._gen_batch_data(batch_size, in_dim, out_dim)
        _, y_ = self._gen_batch_data(batch_size, in_dim, out_dim)

        # Initialize losses
        loss_ours, loss_theirs = self._init_modules()

        # Compute forward pass
        out_ours = loss_ours(y_, y)
        out_theirs = loss_theirs(y_, y)

        assert out_ours.isclose(out_theirs, rtol=thresh).all(), 'Output of loss must be equal'
        return loss_ours, loss_theirs

    def test_forward(self):
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            batch_size = random.randint(1, max_batch_size)
            self._forward(batch_size, in_dim, out_dim)