from unittest import TestCase
import torch, random
from test_module import TestModule
from nn.activation import ReLU, Tanh
from nn.loss import MSELoss
from nn.sequential import Sequential

n_tests = 10
thresh = 1e-3
max_dim = 100
max_batch_size = 100

class TestActivation(TestModule):
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
    
    
    def _forward(self, batch_size, dim, name):
        """
        Forward test function for activation modules
        """
        # Generate random test data
        x, y = self._gen_batch_data(batch_size, dim, dim)
        
        # Initialization
        act_ours, act_theirs = self._init_activations(name)
        
        # Compute forward
        out_ours = act_ours(x)
        out_theirs = act_theirs(x)
        
        # Check output
        assert out_ours.isclose(out_theirs, rtol=thresh).all(), 'Outputs of linear must be the same'
        
        
    def _backward(self, batch_size, in_dim, out_dim, name):
        """
        Backward test function for activation modules
        
        Idea: Put the activation in a Sequential with an Linear Layer and observe backward output.
        This only works if we assume that MSELoss, Sequential and Linear are working.
        """
        
        # Create data
        x, y = self._gen_batch_data(batch_size, in_dim, out_dim)
        
        # Creating sequential
        module_ours, module_theirs = self._init_modules(in_dim, out_dim)
        if name == 'relu':
            seq_ours = Sequential((module_ours, ReLU()), MSELoss())
            seq_theirs = torch.nn.Sequential(module_theirs, torch.nn.ReLU())
        elif name == 'tanh':
            seq_ours = Sequential((module_ours, Tanh()), MSELoss())
            seq_theirs = torch.nn.Sequential(module_theirs, torch.nn.Tanh())
        else:
            raise NotImplementedError
                
        # Forward 
        out_ours = seq_ours(x)
        out_theirs = seq_theirs(x)
        
        # Loss
        loss_ours = seq_ours.loss(out_ours, y)
        loss_fn_theirs = torch.nn.MSELoss()
        loss_theirs = loss_fn_theirs(out_theirs, y)
        
        # Backward
        seq_ours.backward()
        loss_theirs.backward()
        
        # Get modules back
        m_ours, m_theirs = seq_ours.modules, list(seq_theirs.children())[0]
        
        # Get gradients
        dw_ours = m_ours[0].cache['dw_glob']
        dw_theirs = m_theirs.weight.grad
        
        db_ours = m_ours[0].cache['db_glob']
        db_theirs = m_theirs.bias.grad
        
        if db_ours.isclose(db_theirs, rtol=thresh).all() == False:
            print(dw_ours)
            print(dw_theirs)
        
        assert db_ours.isclose(db_theirs, rtol=thresh).all(), 'Gradients of the bias must be equal'
        assert dw_ours.isclose(dw_theirs, rtol=thresh).all(), 'Gradients of the weights must be equal'
        
        
    def test_relu_forward(self):
        """
        Testing relu activation forward
        """
        for _ in range(n_tests):
            dim = random.randint(1, max_dim)
            batch_size = random.randint(1, max_batch_size)
            self._forward(batch_size, dim, 'relu')


    def test_tanh_forward(self):
        """
        Testing tanh activation forwards
        """
        for _ in range(n_tests):
            dim = random.randint(1, max_dim)
            batch_size = random.randint(1, max_batch_size)
            self._forward(batch_size, dim, 'tanh')

    
    def test_relu_backward(self):
        """
        Testing relu activation backwards, can only work if sequential, linear and MSELoss
        are working too.
        """
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            batch_size = random.randint(1, max_batch_size)
            self._backward(batch_size, in_dim, out_dim, 'relu')
            
            
    def test_tanh_backward(self):
        """
        Testing tanh activation backwards, can only work if sequential, linear and MSELoss
        are working too.
        """
        for _ in range(n_tests):
            in_dim = random.randint(1, max_dim)
            out_dim = random.randint(1, max_dim)
            batch_size = random.randint(1, max_batch_size)
            self._backward(batch_size, in_dim, out_dim, 'tanh')
            