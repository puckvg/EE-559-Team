from src.utils import load_target_data, load_all_data,  
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
                    action='store_true', default=25,
                    help = 'Number of training epochs (default=25)')
parser.add_argument('--n_cv', 
                    action='store_true', default=1,
                    help = 'Number of cross-validation folds (default=1)')
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
               'pretrained_tail']

def cv_train(name, model, args, run):
    """ Helper function for cross-valdiated training """
    train_acc, val_acc, test_acc = [], [], []
    for i in range(args.n_cv):
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


if args.train:
    # Training of models
    
    ### m1 ###
    m1_train_acc, m1_val_acc, m1_test_acc = [], [], []
    for i in range(args.n_cv):
        # Define, train and evaluate model
        dl_train_target, dl_val_target, dl_test_target = load_target_data(normalize=True)
        trainer = Trainer(nb_epochs=args.nb_epochs, verbose=False, run='fc_baseline')
        m1 = LinearBaseline()
        _, acc_train, acc_val = trainer.fit(m1, dl_train_target, dl_val_target, verbose=True)
        acc_test = trainer.test(m1, dl_test_target, test_verbose=False, return_acc=True)
        m1_train_acc.append(acc_train)
        m1_val_acc.append(acc_val)
        m1_test_acc.append(acc_test) 
    if args.save_models:
        torch.save(m1.state_dict(), 'models/m1.pt')
    
    
    ### m2 ###
    alpha = LinearAlpha()
    beta = LinearBeta(label_encoded=False)
    m2 = Siamese(alpha, beta, weight_aux=0.5, softmax=False, argmax=False, strategy='sum')
    m2_train_acc, m2_val_acc, m2_test_acc = cv_train('m2', m2, args, 'fc_aux_argmax') 
    
    
    ### m3 ###
    le_net = LeNet()
    linear = nn.Linear(20, 2)
    m3 = Siamese(le_net, target=linear, weight_aux=0., strategy="sum",
                            softmax=False, argmax=False)
    m3_train_acc, m3_val_acc, m3_test_acc = cv_train('m3', m3, args, 'conv_no_aux')
    
    
    ### m4 ###
    tail_net = nn.Linear(20,2)
    m4 = Siamese(le_net, softmax=False, argmax=False, strategy="sum", 
                                    target=tail_net, weight_aux=0.8)
    m4_train_acc, m4_val_acc, m4_test_acc = cv_train('m4', m4, args, 'conv_aux')
        
    
    ### m5 ###
    le_net = LeNet()
    m5 = Siamese(le_net, target=None, softmax=False, argmax=False, strategy="sum", 
                            weight_aux=0.)
    m5_train_acc, m5_val_acc, m5_test_acc = cv_train('m5', m5, args, 'digit')
    
    
    ### m6 ###
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m6 = Siamese(le_net, target=tail_net, weight_aux=0.6, softmax=False,
                        argmax=False, strategy='sum')
    m6_train_acc, m6_val_acc, m6_test_acc = cv_train('m6', m6, args, 'conv_aux_tail')
    
    
    ### m7 ###
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m7 = Siamese(le_net, target=tail_net, weight_aux=0.6, strategy='sum',
                        softmax=True, argmax=False)
    m7_train_acc, m7_val_acc, m7_test_acc = cv_train('m7', m7, args, 'conv_softmax')
    
    
    ### m8 ###
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=True)
    m8 = Siamese(le_net, target=tail_net, weight_aux=0.6, strategy='sum',
                            softmax=False, argmax=True)
    m8_train_acc, m8_val_acc, m8_test_acc = cv_train('m8', m8, args, 'conv_argmax')
    
    
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
    
    
else:
    # Load models from pickel files
    
    ### m1 ###
    m1 = LinearBaseline()
    m1.load_state_dict(torch.load('models/m1.pt'))
    
    ### m2 ###
    alpha = LinearAlpha()
    beta = LinearBeta(label_encoded=False)
    m2 = Siamese(alpha, beta, weight_aux=0.5, softmax=False, 
                    argmax=False, strategy='sum')
    m2.load_state_dict(torch.load('models/m2.pt'))
    
    ### m3 ###
    le_net = LeNet()
    linear = nn.Linear(20, 2)
    m3 = Siamese(le_net, target=linear, weight_aux=0., strategy="sum",
                            softmax=False, argmax=False)
    m3.load_state_dict(torch.load('models/m3.pt'))
    
    ### m4 ###
    tail_net = nn.Linear(20,2)
    m4 = Siamese(le_net, softmax=False, argmax=False, strategy="sum", 
                                    target=tail_net, weight_aux=0.8)
    m4.load_state_dict(torch.load('models/m4.pt'))
    
    ### m5 ###
    le_net = LeNet()
    m5 = Siamese(le_net, target=None, softmax=False,
                            argmax=False, strategy="sum", 
                            weight_aux=0.)
    m5.load_state_dict(torch.load('models/m5.pt'))
    
    ### m6 ###
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m6 = Siamese(le_net, target=tail_net, weight_aux=0.6, softmax=False,
                        argmax=False, strategy='sum')
    m6.load_state_dict(torch.load('models/m6.pt'))
    
    ### m7 ###
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m7 = Siamese(le_net, target=tail_net, weight_aux=0.6, strategy='sum',
                        softmax=True, argmax=False)
    m7.load_state_dict(torch.load('models/m7.pt'))
    
    ### m8 ###
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=True)
    m8 = Siamese(le_net, target=tail_net, weight_aux=0.6, strategy='sum',
                            softmax=False, argmax=True)
    m8.load_state_dict(torch.load('models/m8.pt'))
    
    ### m9 ###
    tail = TailLinear(label_encoded=True)
    le_net = LeNet()
    m9 = Siamese(auxiliary=le_net, target=tail, softmax=False, strategy='sum')
    m9.load_state_dict(torch.load('models/m9.pt'))
    
        

##############################################################################
### Evaluation

if args.train:
    # Printing after training
    print("\n####### Train results #######")
    print("\nNumber of folds: {:2d}\n".format(args.n_cv))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[0], np.array(m1_test_acc).mean(), np.array(m1_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[1], np.array(m2_test_acc).mean(), np.array(m2_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[2], np.array(m3_test_acc).mean(), np.array(m3_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[3], np.array(m4_test_acc).mean(), np.array(m4_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[4], np.array(m5_test_acc).mean(), np.array(m5_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[5], np.array(m6_test_acc).mean(), np.array(m6_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[6], np.array(m7_test_acc).mean(), np.array(m7_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[7], np.array(m8_test_acc).mean(), np.array(m8_test_acc).std()))
    print("{:30s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[8], np.array(m9_test_acc).mean(), np.array(m9_test_acc).std()))
else:
    # Evaluating models on 5 data samples
    print("\n####### Evaluation results #######")
    print("\nNumber of folds: 5\n")
    
    _, _, dl_test_1 = load_all_data(normalize=True)
    _, _, dl_test_2 = load_all_data(normalize=True)
    _, _, dl_test_3 = load_all_data(normalize=True)
    _, _, dl_test_4 = load_all_data(normalize=True)
    _, _, dl_test_5 = load_all_data(normalize=True)
    dl_test = [dl_test_1, dl_test_2, dl_test_3, dl_test_4, dl_test_5]
    
    _, _, dl_test_t_1 = load_target_data(normalize=True)
    _, _, dl_test_t_2 = load_target_data(normalize=True)
    _, _, dl_test_t_3 = load_target_data(normalize=True)
    _, _, dl_test_t_4 = load_target_data(normalize=True)
    _, _, dl_test_t_5 = load_target_data(normalize=True)
    dl_test_t = [dl_test_t_1, dl_test_t_2, dl_test_t_3, dl_test_t_4, dl_test_t_5]
    
    models = [m2, m3, m4, m5, m6, m7, m8, m9]
    
    trainer = Trainer(nb_epochs=0)
    
    # Model 1
    test_acc = []
    for dl in dl_test_t:
        test_acc.append(trainer.test(m1, dl, test_verbose=False, return_acc=True))
    print("{:26s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[0], np.array(test_acc).mean(), np.array(test_acc).std()))
    
    # Model 2-9
    for i, m in enumerate(models):
        test_acc = []
        for dl in dl_test:
            test_acc.append(trainer.test(m, dl, test_verbose=False, return_acc=True))
        print("{:26s}: Test accuracy: {:1.2f} ({:2.2f})".format(model_names[i+1], np.array(test_acc).mean(), np.array(test_acc).std()))









