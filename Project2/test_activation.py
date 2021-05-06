from unittest import TestCase
import torch, random
from test_module import TestModule
from nn.activation import *

n_tests = 10
thresh = 1e-3

class TestActivation(TestModule):
    def _forward(self, dim, name):
        """
        Forward test function for activation modules
        """
        # Generate random test data
        x, _ = self._gen_data(dim, dim)
        
        # Initialization
        act_ours, act_theirs = self._init_activations(name)
        
        # Compute forward
        out_ours = act_ours(x)
        out_theirs = act_theirs(x)
        
        # Check output
        assert out_ours.isclose(out_theirs, rtol=thresh).all(), 'Outputs of linear must be the same'
        return out_ours, out_theirs
        
    def _backward(self, dim, name):
        """
        Backward test function for activation modules
        """
        # Generate random test data
        x, _ = self._gen_data(dim, dim)
        
        # Initialization
        act_ours, act_theirs = self._init_activations(name)
        
        
        
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




testactivation = TestActivation()
testactivation._forward(dim = 3, name='relu')
