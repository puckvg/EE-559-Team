from src.dlc_practical_prologue import generate_pair_sets
from src.utils import load_class_data, load_target_data, load_all_data, print_param_count, plot_training_epochs, multi_plot_training_epochs
from src.models import *
from src.trainer import Trainer

import numpy as np
import matplotlib.pyplot as plt 

import argparse
import os

# Getting current working directory
cwd = os.getcwd()

##############################################################################
### Getting input arguments

parser = argparse.ArgumentParser(
    description='Test file for project 1. Can do everything.')
parser.add_argument('--train', 
                    action='store_true', default=False,
                    help= '''Train models. If False, models will 
                    be loaded. (default=False)''')
parser.add_argument('--nb_epochs',
                    action='store_true', default=25,
                    help = 'Number of training epochs (default=25)')
parser.add_argument('--n_cv', 
                    action='store_true', default=1,
                    help = 'Number of cross-validation folds (default=1)')
parser.add_argument('--save_models', 
                    action='store_true', default=False,
                    help = '''Save trained models in pickle files. Be careful: 
                    previous saves are overwritten. (default=False)''')

args = parser.parse_args()

##############################################################################
### Training / loading

"""
Nomenclature used:
m1      1-fc_baseline
m2      fc_aux
m3      3-conv_no_aux
m4      4-conv_aux
m5      5-conv_aux_tailnet
m6      6-digits_to_class
m7      7-conv_aux_tailnet_softmax
m8      8-conv_aux_tailnet_argmax
m9      9-pretrained_tail
m10     10-pretrained_seperatly
"""


if args.train:
    # Training of models
    
    ### m1 ###
    m1_train_acc, m1_val_acc, m1_test_acc = [], [], []
    for i in range(args.n_cv):
        # Define, train and evaluate model
        dl_train_target, dl_val_target, dl_test_target = load_target_data(normalize=True)
        trainer = Trainer(nb_epochs=args.nb_epochs, verbose=False, run='fc_baseline')
        m1 = LinearBaseline()
        _, acc_train, acc_val = trainer.fit(m1, dl_train_target, dl_val_target, verbose=False)
        acc_test = trainer.test(m1, dl_test_target, test_verbose=False, return_acc=True)
        m1_train_acc.append(acc_train)
        m1_val_acc.append(acc_val)
        m1_test_acc.append(acc_test) 
    if args.save_models:
        print(cwd+"models/m1.pt")
        torch.save(m1.state_dict(), cwd+"models/m1.pt")
    
    ### m2 ###
    m2_train_acc, m2_val_acc, m2_test_acc = [], [], []
    for i in range(args.n_cv):
        # Define, train and evaluate model
        dl_train_all, dl_val_all, dl_test_all = load_all_data(normalize=True)
        trainer = Trainer(nb_epochs=args.nb_epochs, verbose=False, run='fc_baseline')
        alpha = LinearAlpha()
        beta = LinearBeta(label_encoded=False)
        m2 = Siamese(alpha, beta, weight_aux=0.5, softmax=False, 
                    argmax=False, strategy='sum')
        _, acc_train, acc_val = trainer.fit(m2, dl_train_all, dl_val_all, verbose=True)
        acc_test = trainer.test(m2, dl_test_all, test_verbose=False, return_acc=True)
        m1_train_acc.append(acc_train)
        m1_val_acc.append(acc_val)
        m1_test_acc.append(acc_test)
    if args.save_models:
        torch.save(m2.state_dict(), cwd + "models/m2.pt")  
    
    
else:
    # Load models from pickel files
    pass

##############################################################################
### Evaluation
