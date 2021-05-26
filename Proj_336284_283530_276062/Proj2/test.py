import torch
from torch import empty
import argparse

from nn.activation import ReLU
from nn.linear import Linear
from nn.loss import MSELoss
from nn.sequential import Sequential
from trainer import Trainer

# Turn autograd off
torch.set_grad_enabled(False)

# -----------------------------------------------------
#                     Parameters
# -----------------------------------------------------

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="""Test script for Project 2""",
)

parser.add_argument(
    "--batch_size",
    action="store",
    default=64,
    type=int,
    help="""Batch size for training and testing. (default=64)""",
)

parser.add_argument(
    "--nb_epochs",
    action="store",
    default=100,
    type=int,
    help="""Number of training epochs. (default=100)""",
)

parser.add_argument(
    "--n_samples",
    action="store",
    default=1000,
    type=int,
    help="""Number of samples in training and test set. (default=1000)""",
)

parser.add_argument(
    "--n_folds",
    action="store",
    default=1,
    type=int,
    help="""Number of random initializations. (default=1)""",
)

args = parser.parse_args()


# -----------------------------------------------------
#                    Creating data
# -----------------------------------------------------


def gen_data(n):
    x = empty((2 * n, 2)).uniform_(0, to=1)
    pi = empty((1)).fill_(0).acos().item() * 2

    target = ((x - empty(1, 2).fill_(0.5)).pow(2).sum(dim=1) <= 1 / (2 * pi)) * 1

    x_train, x_test = x[:n], x[n:]
    y_train, y_test = target[:n], target[n:]
    return x_train, x_test, y_train.view(-1, 1), y_test.view(-1, 1)


train_error_rate_SGD, test_error_rate_SGD = [], []
train_error_rate_Adam, test_error_rate_Adam = [], []

for i in range(args.n_folds):
    x_train, x_test, y_train, y_test = gen_data(n=args.n_samples)

    # -----------------------------------------------------
    #                    Creating model
    # -----------------------------------------------------

    LinNet_SGD = Sequential(
        (
            Linear(2, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 1),
        ),
        MSELoss(),
    )

    LinNet_Adam = Sequential(
        (
            Linear(2, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 25),
            ReLU(),
            Linear(25, 1),
        ),
        MSELoss(),
    )

    if i==0:
        print("\n### Model structure: ")
        LinNet_SGD.print()


    # -----------------------------------------------------
    #                      Training
    # -----------------------------------------------------

    trainer = Trainer(nb_epochs=args.nb_epochs)

    verbose = i == 0

    if i==0: print("\n### Training using SGD:")
    _ = trainer.fit(
        LinNet_SGD,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=args.batch_size,
        lr=0.1,
        verbose=verbose,
        print_every=10,
        optim="sgd",
    )

    if i==0: print("\n### Training using Adam:")
    _ = trainer.fit(
        LinNet_Adam,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=args.batch_size,
        lr=0.01,
        verbose=verbose,
        print_every=10,
        optim="adam",
    )


    # -----------------------------------------------------
    #                      Evaluation
    # -----------------------------------------------------

    # SGD
    train_pred = LinNet_SGD(x_train).round()
    test_pred = LinNet_SGD(x_test).round()

    train_error_rate = (train_pred != y_train).sum().item() / args.n_samples
    test_error_rate = (test_pred != y_test).sum().item() / args.n_samples

    train_error_rate_SGD.append(train_error_rate)
    test_error_rate_SGD.append(test_error_rate)

    # Adam
    train_pred = LinNet_Adam(x_train).round()
    test_pred = LinNet_Adam(x_test).round()

    train_error_rate = (train_pred != y_train).sum().item() / args.n_samples
    test_error_rate = (test_pred != y_test).sum().item() / args.n_samples
    
    train_error_rate_Adam.append(train_error_rate)
    test_error_rate_Adam.append(test_error_rate)

    if i == 0:
        print("\n### Evaluation")
        print("          |            SGD           |           Adam           |")
        print("Iteration | Train error | Test error | Train error | Test error |")
        
    print("{:9d} | {:10.2f}% |  {:8.2f}% |  {:9.2f}% |  {:8.2f}% |".format(i+1, train_error_rate_SGD[-1] * 100, test_error_rate_SGD[-1] * 100, train_error_rate_Adam[-1] * 100, test_error_rate_Adam[-1] * 100))
    
if args.n_folds != 1:
    train_av_SGD = sum(train_error_rate_SGD) / args.n_folds
    test_av_SGD = sum(test_error_rate_SGD) / args.n_folds
    train_av_Adam = sum(train_error_rate_Adam) / args.n_folds
    test_av_Adam = sum(test_error_rate_Adam) / args.n_folds
    
    print("----------|--------- Average result over {:2d} iterations ---------|".format(args.n_folds))
    print("          | {:10.2f}% |  {:8.2f}% |  {:9.2f}% |  {:8.2f}% |".format(train_av_SGD * 100, test_av_SGD * 100, train_av_Adam * 100, test_av_Adam * 100))