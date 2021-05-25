import torch


class Trainer:
    def __init__(self, nb_epochs):
        """Create a trainer by specifying the number of epochs to train.

        Args:
            nb_epochs (int): Number of epochs to train
            verbose (bool): Whether or not to output training information.

        """
        self.nb_epochs = nb_epochs

    def fit(
        self,
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size=32,
        lr=0.01,
        optim="sgd",
        verbose=True,
        print_every=32,
    ):
        """Train the model on the specified data and print the training and validation loss and accuracy.

        Args:
            model (nn.Module): Model to train
            x_train (torch.tensor): Training data
            y_train (torch.tensor): Training labels 
            x_val (torch.tensor): Validation data 
            y_val (torch.tensor): Validation labels
            batch_size (int): Batch sizes for training and validation 
            lr (float): Learning rate for optimization (Default is 0.01)
            optim (str): Optimizer (options are 'sgd' or 'adam'. Default is 'sgd')
            verbose (bool): Whether or not to output training information (Default is True)
            print_every (str): How often to print progress (Default is every 32 steps)

        Example:
            Use the trainer to fit a new nn model.

            >>> from trainer import Trainer
            >>> from torch import empty
            >>> from nn.sequential import Sequential
            >>> ... # Read data into x_train, y_train, x_test, y_test
            >>> LinNet = LinNet = Sequential((Linear(2, 1), MSELoss())
            >>> trainer = Trainer(nb_epochs=25)
            >>> loss_train, loss_val = trainer.fit(LinNet, x_train, y_train, x_test, y_test, batch_size=32, lr=0.1, print_every=10, optim='sgd')
        """

        train_loss_epochs = []
        val_loss_epochs = []

        for e in range(self.nb_epochs):
            loss_train = []
            for batch in range(0, len(x_train), batch_size):
                x_batch = x_train[batch : batch + batch_size]
                y_batch = y_train[batch : batch + batch_size]

                loss = model.training_step(x_batch, y_batch)
                model.backward()
                model.update_params(optim=optim, lr=lr)

                loss_train.append(loss.item())

            loss_val = []

            if verbose:
                for batch in range(0, len(x_val), batch_size):
                    x_batch = x_val[batch : batch + batch_size]
                    y_batch = y_val[batch : batch + batch_size]

                    loss = model.validation_step(x_batch, y_batch)
                    loss_val.append(loss.item())

                avg_loss_train = round(sum(loss_train) / len(loss_train), 2)
                train_loss_epochs.append(avg_loss_train)

                avg_loss_val = round(sum(loss_val) / len(loss_val), 2)
                val_loss_epochs.append(avg_loss_val)
                if (e % print_every == 0) or (e + 1 == self.nb_epochs):
                    print(
                        "# Epoch {:3d}/{:d}:\t loss={:10.4e}\t loss_val={:10.4e}".format(
                            e + 1, self.nb_epochs, avg_loss_train, avg_loss_val
                        )
                    )

        if verbose:
            return train_loss_epochs, val_loss_epochs

    def test(self, model, x_test, y_test, batch_size=32, test_verbose=True):
        """Test the model on the specified data.

        Args:
            model (nn.Module): Model to train
            x_test (torch.tensor): Test data 
            y_test (torch.tensor): Test labels 
            batch_size (int): Batch size for testing
            test_verbose (bool): Whether the test result should be printed

        Example:
            Use the trainer to test an existing nn model.

            >>> from trainer import Trainer
            >>> from torch import empty
            >>> ... # Train model LinNet
            >>> ... # Read data into x_train, y_train, x_test, y_test
            >>> trainer = Trainer(nb_epochs=25)
            >>> loss_test = t.test(LinNet, x_test, y_test, batch_size=32, test_verbose=True)
            loss_test=0.17

        """

        loss_test = []
        for batch in range(0, len(x_test), batch_size):
            x_batch = x_test[batch : batch + batch_size]
            y_batch = y_test[batch : batch + batch_size]

            loss = model.test_step(x_batch, y_batch)
            loss_test.append(loss.item())

        avg_loss_test = round(sum(loss_test) / len(loss_test), 2)
        if test_verbose:
            print(f"loss_test={avg_loss_test}")
        if test_verbose:
            return avg_loss_test
