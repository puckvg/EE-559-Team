from torch import empty

from nn.sequential import Sequential
from nn.activation import ReLU
from nn.linear import Linear
from nn.loss import MSELoss
from trainer import Trainer

# -----------------------------------------------------
#                    Creating data 
# -----------------------------------------------------
print("Creating data...")

def gen_data(n):
    x = empty((2 * n, 2)).random_()
    pi = empty((1)).acos().item() * 2
    target = ((x - empty(1,2).fill_(0.5)).pow(2).sum(dim=1) <= 1/(2*pi)) * 1

    x_train, x_test = x[:n], x[n:]
    y_train, y_test = target[:n], target[n:]
    return x_train, x_test, y_train.view(-1, 1), y_test.view(-1, 1)

x_train, x_test, y_train, y_test = gen_data(n = 1000)

# -----------------------------------------------------
#                    Creating model 
# -----------------------------------------------------
print("Creating model...")

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
print("Training...")

trainer = Trainer(nb_epochs=25)

_ = trainer.fit(LinNet, x_train, y_train, x_test, y_test, batch_size=32, optim='sgd')







