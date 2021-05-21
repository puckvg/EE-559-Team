from torch import empty

from nn.sequential import Sequential
from nn.activation import ReLU
from nn.linear import Linear
from nn.loss import MSELoss
from trainer import Trainer

# -----------------------------------------------------
#                     Parameters 
# -----------------------------------------------------

batch_size = 32
nb_epochs = 300
n_samples = 1000
print_every = 10


# -----------------------------------------------------
#                    Creating data 
# -----------------------------------------------------

def gen_data(n):
    x = empty((2 * n, 2)).uniform_(0, to=1)
    pi = empty((1)).fill_(0).acos().item() * 2

    target = ((x - empty(1,2).fill_(0.5)).pow(2).sum(dim=1) <= 1/(2*pi)) * 1

    x_train, x_test = x[:n], x[n:]
    y_train, y_test = target[:n], target[n:]
    return x_train, x_test, y_train.view(-1, 1), y_test.view(-1, 1)

x_train, x_test, y_train, y_test = gen_data(n_samples)


# -----------------------------------------------------
#                    Creating model 
# -----------------------------------------------------

def init_model(dim_in, dim_out, dim_hidden, n_hidden=3):
    net = Sequential((
        Linear(dim_in, dim_hidden),
        ReLU(),
        *(Linear(dim_hidden, dim_hidden), ReLU()) * (n_hidden),
        Linear(dim_hidden, dim_out)),
        MSELoss()
    )
    return net

LinNet = init_model(dim_in=2, dim_out=1, dim_hidden=25, n_hidden=3)

print("\n### Model structure:")
LinNet.print()

# -----------------------------------------------------
#                      Training 
# -----------------------------------------------------

t = Trainer(nb_epochs)

print("\n### Training:")
_ = t.fit(LinNet, x_train, y_train, x_test, y_test, batch_size=batch_size, print_every=print_every, optim='sgd', lr=0.02)

# -----------------------------------------------------
#                      Evaluation 
# -----------------------------------------------------

y_train_pred = LinNet(x_train).round()
y_test_pred = LinNet(x_test).round()

train_error = (y_train_pred != y_train).sum() / n_samples * 100
test_error = (y_test_pred != y_test).sum() / n_samples * 100

print("\nFinal train error: {:6.2f}%".format(train_error))
print("Final test error: {:6.2f}%".format(test_error))




