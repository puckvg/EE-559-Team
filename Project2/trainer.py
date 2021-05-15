import torch

class Trainer:
    def __init__(self, nb_epochs):
        """ Create a trainer by specifying the number of epochs to train 
        Args:
            nb_epochs: int. Number of epochs to train
            verbose: bool. Whether or not to output training information.
        """
        self.nb_epochs = nb_epochs
    
    def fit(self, model, x_train, y_train, x_val, y_val, batch_size=32, lr=0.01, optim='sgd', verbose=True):
        """ Train the model on the specified data and print the training and validation loss and accuracy.
        Args:
            model: Module. Model to train
            dl_train: DataLoader. DataLoader containing the training data
            dl_val: DataLoader. DataLoader containting the validation data
            verbose: bool. Whether or not to output training information
        """

        train_loss_epochs = []
        val_loss_epochs = []

        for e in range(self.nb_epochs):
            loss_train = []
            for batch in range(0, len(x_train), batch_size):
                x_batch = x_train[batch:batch+batch_size]
                y_batch = y_train[batch:batch+batch_size]

                loss = model.training_step(x_batch, y_batch)
                model.backward()
                model.update_params(optim=optim, lr=lr)

                loss_train.append(loss.item())

            loss_val = []

            if verbose:
                for batch in range(0, len(x_val), batch_size):
                    x_batch = x_val[batch:batch+batch_size]
                    y_batch = y_val[batch:batch+batch_size]

                    loss = model.validation_step(x_batch, y_batch)
                    loss_val.append(loss.item())

                avg_loss_train = round(sum(loss_train)/len(loss_train), 2)
                train_loss_epochs.append(avg_loss_train)

                avg_loss_val = round(sum(loss_val)/len(loss_val), 2)
                val_loss_epochs.append(avg_loss_val)

                if (e) % batch_size == 0:
                    print(f'# Epoch {e+1}/{self.nb_epochs}:\t loss={avg_loss_train}\t loss_val={avg_loss_val}')

        if verbose:
            return train_loss_epochs, val_loss_epochs

    def test(self, model, x_test, y_test, batch_size=32, test_verbose=True):
        """ Test the model on the specified data 
        Args:
            model: Module. Model to train
            dl_test: DataLoader. DataLoader containting the test data
            test_verbose: bool. Whether the test result should be printed
        """

        loss_test = []
        acc_test = []
        for batch in range(0, len(x_test), batch_size):
            x_batch = x_test[batch:batch+batch_size]
            y_batch = y_test[batch:batch+batch_size]

            loss, acc = model.test_step(x_batch, y_batch)
            loss_test.append(loss.item())
            acc_test.append(acc)

        avg_loss_test = round(sum(loss_test)/len(loss_test), 2)
        avg_acc_test = round(sum(acc_test)/len(acc_test), 2)
        if test_verbose:
            print(f'loss_test={avg_loss_test}\t acc_test={avg_acc_test}')
        if test_verbose: 
            return avg_acc_test
