# Project 1

A number of models are available to predict, given a series of 2x14x14 tensors corresponding to pairs of 14x14 grayscale images, whether the first digit is lesser or equal to the second.

The data is loaded using `dlc_practical_prologue.py`, the models are available in `models.py`, and `trainer.py` facilitates the training of the models. 
The `test.py` file generates or loads the saved models in `models` (the default) and trains the models in the case of the former and tests the models on unseen pairs of images for both
and prints the output to the screen. 

`test.py` can be run with arguments as follows: 
```
python test.py --train --nb_epochs --n_cv --save_models
```
where the defaults are `train=False`, `nb_epochs=25`, `n_cv=10`, `save_models=False`. 

