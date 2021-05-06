from unittest import TestCase
import torch, random
from test_module import TestModule
from nn.activation import *
from nn.loss import *

n_tests = 10
thresh = 1e-3
max_dim = 100

class TestActivation(TestModule):
    def _forward(self, dim, name):
        """
        Forward test function for activation modules
        """
        # Generate random test data
        x, y = self._gen_data(dim, dim)
        
        # Initialization
        act_ours, act_theirs = self._init_activations(name)
        
        # Compute forward
        out_ours = act_ours(x)
        out_theirs = act_theirs(x)
        
        # Check output
        assert out_ours.isclose(out_theirs, rtol=thresh).all(), 'Outputs of linear must be the same'
        return x, y, out_ours, out_theirs, act_ours, act_theirs
        
        
    def test_relu_forward(self):
        """
        Testing relu activation forward
        """
        for _ in range(n_tests):
            dim = random.randint(1, max_dim)
            self._forward(dim, 'relu')


    def test_tanh_forward(self):
        """
        Testing tanh activation forwards
        """
        for _ in range(n_tests):
            dim = random.randint(1, max_dim)
            self._forward(dim, 'tanh')


    def _backward(self, dim, name):
        """
        Backward test function for activation modules
        """
        # Forward pass
        x, y, out_ours, out_theirs, act_ours, act_theirs = self._forward(dim, name)
        
        # Initialize losses
        loss_fn_ours = MSELoss()
        loss_fn_theirs = torch.nn.MSELoss()
        
        # Calculate losses
        loss_ours = loss_fn_ours(out_ours, y)
        loss_theirs = loss_fn_theirs(out_theirs, y)
        
        # Our backwards
        dy = loss_fn_ours.backward()
        act_ours.backward(dy)
        
        # Their backwards
        loss_theirs.backward()
        
        # Compare
        "BUT HOW???"
        
        
    def _init_activations(self, name):
        """
        Initializing activation function modules
        """
        if name == 'relu':
            act_ours = ReLU()
            act_theirs = torch.nn.ReLU()
        elif name == 'tanh':
            act_ours = Tanh()
            act_theirs = torch.nn.Tanh()
        else:
            raise NotImplementedError
        return act_ours, act_theirs

