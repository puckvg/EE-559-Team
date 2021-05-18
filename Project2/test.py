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
nb_epochs = 100


# -----------------------------------------------------
#                    Creating data 
# -----------------------------------------------------

def gen_data(n):
    x = empty((2 * n, 2)).normal_()
    pi = empty((1)).fill_(0).acos().item() * 2

    target = ((x - empty(1,2).fill_(0.5)).pow(2).sum(dim=1) <= 1/(2*pi)) * 1

    x_train, x_test = x[:n], x[n:]
    y_train, y_test = target[:n], target[n:]
    return x_train, x_test, y_train.view(-1, 1), y_test.view(-1, 1)

x_train, x_test, y_train, y_test = gen_data(n = 1000)




# -----------------------------------------------------
#                    Creating model 
# -----------------------------------------------------

LinNet = Sequential((
    Linear(2, 25),
    ReLU(),
    Linear(25, 25),
    ReLU(), 
    Linear(25, 25),
    ReLU(),
    Linear(25, 1)),
    MSELoss()
)

# -----------------------------------------------------
#                      Training 
# -----------------------------------------------------

trainer = Trainer(nb_epochs=nb_epochs)

_ = trainer.fit(LinNet, x_train, y_train, x_test, y_test, batch_size=batch_size, optim='adam')







