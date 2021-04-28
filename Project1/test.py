from src.utils import load_target_data, load_all_data
from src.models import *
from src.trainer import Trainer

import numpy as np

import argparse


##############################################################################
### Getting input arguments

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=  
    '''Test file for project 1. Can do everything :)
    
    If --train is not given, trained models will be loaded and evaluated.
    
    If --train argument is given, n_cv trainings from different random 
    initializations and data sets are done. (Similar to what has been done 
    in the report)''')

parser.add_argument('--train', 
                    action='store_true', default=False,
                    help= '''Train models. If False, models will be loaded. (default=False)''')
parser.add_argument('--nb_epochs',
                    action='store', default=25, type=int,
                    help = 'Number of training epochs (default=25)')
parser.add_argument('--n_cv', 
                    action='store', default=10, type=int,
                    help = 'Number of random reinitializations (default=10)')
parser.add_argument('--save_models', 
                    action='store_true', default=False,
                    help = '''Save trained models in pickle files. Be careful: previous saves are overwritten. (default=False)''')

args = parser.parse_args()

##############################################################################
### Training / loading

"""
Nomenclature used:
m1      1-fc_baseline
m2      2-fc_aux
m3      3-conv_net
m4      4-conv_aux
m5      5-digits
m6      6-conv_tail
m7      7-conv_softmax
m8      8-conv_argmax
m9      9-pretrained_tail
m10     10-pretrain_random
"""

model_names = ['fc_baseline', 'fc_aux', 'conv_net', 'conv_aux', 'digits',
               'conv_tail', 'conv_softmax', 'conv_argmax',
               'pretrain_tail', 'pretrain_random']

def cv_train(name, model_id, args, run):
    """ Helper function for cross-valdiated training """
    train_acc, val_acc, test_acc = [], [], []
    for _ in range(args.n_cv):
        model = init_model(model_id)
        # Define, train and evaluate model
        dl_train_all, dl_val_all, dl_test_all = load_all_data(normalize=True)
        trainer = Trainer(nb_epochs=args.nb_epochs, verbose=False, run=run)
        _, acc_train, acc_val = trainer.fit(model, dl_train_all, dl_val_all, verbose=True)
        acc_test = trainer.test(model, dl_test_all, test_verbose=False, return_acc=True)
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        test_acc.append(acc_test)
    if args.save_models:
        # Save model
        torch.save(model.state_dict(), 'models/' + name + '.pt')
    return train_acc, val_acc, test_acc


# Define models
    
models = []

### m1 ###
def init_m1():
    m1 = LinearBaseline()
    return m1

### m2 ###
def init_m2():
    alpha = LinearAlpha()
    beta = LinearBeta(label_encoded=False)
    m2 = Siamese(alpha, beta, weight_aux=0.5, softmax=False, 
                argmax=False, strategy='sum')
    return m2

### m3 ###
def init_m3():
    le_net = LeNet()
    linear = nn.Linear(20, 2)
    m3 = Siamese(le_net, target=linear, weight_aux=0., strategy="sum",
                            softmax=False, argmax=False)
    return m3

### m4 ###
def init_m4():
    le_net = LeNet()
    tail_net = nn.Linear(20,2)
    m4 = Siamese(le_net, softmax=False, argmax=False, strategy="sum", 
                                target=tail_net, weight_aux=0.8)
    return m4

### m5 ###
def init_m5():
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m5 = Siamese(le_net, target=tail_net, weight_aux=0.6, softmax=False,
                        argmax=False, strategy='sum')
    return m5

### m6 ###
def init_m6():
    le_net = LeNet()
    m6 = Siamese(le_net, target=None, softmax=False,
                            argmax=False, strategy="sum", 
                            weight_aux=0.)
    return m6

### m7 ###
def init_m7():
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m7 = Siamese(le_net, target=tail_net, weight_aux=0.6, strategy='sum',
                    softmax=True, argmax=False)
    return m7

### m8 ###
def init_m8():
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=True)
    m8 = Siamese(le_net, target=tail_net, weight_aux=0.6, strategy='sum',
                            softmax=False, argmax=True)
    return m8

### m9 ###
def init_m9():
    tail = TailLinear(label_encoded=True)
    le_net = LeNet()
    m9 = Siamese(auxiliary=le_net, target=tail, softmax=False, strategy='sum')
    return m9

### m10 ###
def init_m10():
    tail = TailLinear(label_encoded=True)
    le_net = LeNet()
    m10 = Siamese(auxiliary=le_net, target=tail, softmax=False, strategy='random')
    return m10

def init_model(i):
    if i==1: return init_m1()
    if i==2: return init_m2()
    if i==3: return init_m3()
    if i==4: return init_m4()
    if i==5: return init_m5()
    if i==6: return init_m6()
    if i==7: return init_m7()
    if i==8: return init_m8()
    if i==9: return init_m9()
    if i==10: return init_m10()

    raise ValueError(f'model {i} not defined.')

if args.train:
    # Training of models
    
    train_accs, val_accs, test_accs = [], [], []

    ### m1 ###
    m1_train_acc, m1_val_acc, m1_test_acc = [], [], []
    for i in range(args.n_cv):
        # Define, train and evaluate model
        m1 = init_model(1)
        dl_train_target, dl_val_target, dl_test_target = load_target_data(normalize=True)
        trainer = Trainer(nb_epochs=args.nb_epochs, verbose=False, run='fc_baseline')
        _, acc_train, acc_val = trainer.fit(m1, dl_train_target, dl_val_target, verbose=True)
        acc_test = trainer.test(m1, dl_test_target, test_verbose=False, return_acc=True)
        m1_train_acc.append(acc_train)
        m1_val_acc.append(acc_val)
        m1_test_acc.append(acc_test) 
    if args.save_models:
        torch.save(m1.state_dict(), 'models/m1.pt')
    
    train_accs.append(m1_train_acc)
    val_accs.append(m1_val_acc)
    test_accs.append(m1_test_acc)
    
    
    ### m2-m8 ###
    for i in range(2, 9, 1):
        train_acc, val_acc, test_acc = cv_train(f'm{i+1}', i+1, args, model_names[i])
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    
    
    ### m9 ###
    m9_train_acc, m9_val_acc, m9_test_acc = [], [], []
    for i in range(args.n_cv):
        # Define, train and evaluate model
        dl_train_all, dl_val_all, dl_test_all = load_all_data(normalize=True)
        
        tail = TailLinear(label_encoded=True)
        trainer = Trainer(nb_epochs=5, run='pretrained_tail_tailTraining')
        trainer.fit(tail, dl_train_all, dl_val_all)
        
        tail.requires_grad_=False
        le_net = LeNet()
        m9 = Siamese(auxiliary=le_net, target=tail, softmax=False, strategy='sum')
        trainer = Trainer(nb_epochs=args.nb_epochs - 5, run='pretrained_tail_headTraining')
        trainer.fit(m9, dl_train_all, dl_val_all)
        
        acc_test = trainer.test(m9, dl_test_all, test_verbose=False, return_acc=True)
        m9_train_acc.append(acc_train)
        m9_val_acc.append(acc_val)
        m9_test_acc.append(acc_test) 
    if args.save_models:
        torch.save(m9.state_dict(), 'models/m9.pt')

    train_accs.append(m9_train_acc)
    val_accs.append(m9_val_acc)
    test_accs.append(m9_test_acc)


    ### m10 ###
    m10_train_acc, m10_val_acc, m10_test_acc = [], [], []
    for i in range(args.n_cv):
        # Define, train and evaluate model
        dl_train_all, dl_val_all, dl_test_all = load_all_data(normalize=True)
        
        tail = TailLinear(label_encoded=True)
        trainer = Trainer(nb_epochs=5, run='pre_train_random_tail')
        trainer.fit(tail, dl_train_all, dl_val_all)
        
        tail.requires_grad_=False
        le_net = LeNet()
        m10 = Siamese(auxiliary=le_net, target=tail, softmax=False, strategy='random')
        trainer = Trainer(nb_epochs=args.nb_epochs - 5, run='pre_train_random_head')
        trainer.fit(m10, dl_train_all, dl_val_all)
        
        acc_test = trainer.test(m9, dl_test_all, test_verbose=False, return_acc=True)
        m10_train_acc.append(acc_train)
        m10_val_acc.append(acc_val)
        m10_test_acc.append(acc_test) 
    if args.save_models:
        torch.save(m10.state_dict(), 'models/m10.pt')
    
    train_accs.append(m10_train_acc)
    val_accs.append(m10_val_acc)
    test_accs.append(m10_test_acc)
    
    
else:
    # Load models from pickel files

    ### m1 - m10 ###
    for i in range(1, 11, 1):
        m = init_model(i)
        m.load_state_dict(torch.load(f'models/m{i}.pt'))
        models.append(m)
    

##############################################################################
### Evaluation

if args.train:
    # Printing after training
    print("\n####### Test results #######")
    print(f"\nAveraged results over {args.n_cv} random reinitializations.\n")
    for name, test_acc in zip(model_names, test_accs):
        print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(name, np.array(test_acc).mean(), np.array(test_acc).std()))
else:
    # Evaluating models
    print("\n####### Test results #######")
    print(f"\nRandom reinitializations:{args.n_cv}\n")
    
    trainer = Trainer(nb_epochs=0)
    
    # Model 1
    test_acc = []
    for _ in range(args.n_cv):
        _, _, dl_test = load_target_data(normalize=True)
        test_acc.append(trainer.test(models[0], dl_test, test_verbose=False, return_acc=True))
    print("{:26s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[0], np.array(test_acc).mean(), np.array(test_acc).std()))
    
    # Model 2-10
    for i, m in enumerate(models[1:]):
        test_acc = []
        for _ in range(args.n_cv):
            _, _, dl_test = load_all_data(normalize=True)
            test_acc.append(trainer.test(m, dl_test, test_verbose=False, return_acc=True))
        print("{:26s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[i+1], np.array(test_acc).mean(), np.array(test_acc).std()))









