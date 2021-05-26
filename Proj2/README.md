# Project 2 
Proj2 is a minimal tensor library for deep learning using CPUs. The current version can:
- Build networks combining fully connected layers, Tanh and RELU
- Run forward and backward passes 
- Optimize parameters with SGD and Adam optimizers for MSE 

The deep learning functionality is in the `nn` module and the `trainer` can be used to facilitate training models.

The `test.py` file first generates 2D toy data of 1,000 train and test points sampled uniformly, with a label 0 if outside a disk centered at (0.5, 0.5) 
of radius 1/sqrt(2pi) and 1 inside. Then it trains a fully connected network with 3 hidden layers, first with the SGD optimizer and then with the Adam optimizer,
and prints the test accuracy to the screen.

The `tests` directory contains unit tests for the `nn` functionality. 
