import torch
from torch import empty

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

batch_size = 64
nb_epochs = 100
n_samples = 1000


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


x_train, x_test, y_train, y_test = gen_data(n=n_samples)


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

print("\n### Model structure: ")
LinNet_SGD.print()


# -----------------------------------------------------
#                      Training
# -----------------------------------------------------

trainer = Trainer(nb_epochs=nb_epochs)

print("\n### Training using SGD:")
_ = trainer.fit(
    LinNet_SGD,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=batch_size,
    lr=0.1,
    print_every=10,
    optim="sgd",
)

print("\n### Training using Adam:")
_ = trainer.fit(
    LinNet_Adam,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=batch_size,
    lr=0.01,
    print_every=10,
    optim="adam",
)


# -----------------------------------------------------
#                      Evaluation
# -----------------------------------------------------

# SGD
train_pred = LinNet_SGD(x_train).round()
test_pred = LinNet_SGD(x_test).round()

train_error_rate = (train_pred != y_train).sum().item() / n_samples
test_error_rate = (test_pred != y_test).sum().item() / n_samples

print("\n### Evaluation")
print("# Results using SGD:")
print("Final train error: {:5.2f}%".format(train_error_rate * 100))
print("Final test error: {:6.2f}%".format(test_error_rate * 100))

# Adam
train_pred = LinNet_Adam(x_train).round()
test_pred = LinNet_Adam(x_test).round()

train_error_rate = (train_pred != y_train).sum().item() / n_samples
test_error_rate = (test_pred != y_test).sum().item() / n_samples

print("\n# Results using Adam:")
print("Final train error: {:5.2f}%".format(train_error_rate * 100))
print("Final test error: {:6.2f}%".format(test_error_rate * 100))