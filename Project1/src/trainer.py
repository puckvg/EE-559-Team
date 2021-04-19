import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, nb_epochs, verbose=True):
        """ Create a trainer by specifying the number of epochs to train """
        self.nb_epochs = nb_epochs
        self.verbose = verbose
    
    def fit(self, model, dl_train, dl_val, verbose=True):
        """ Train the model on the specified data and print the training and validation loss and accuracy.
        Args:
            model: Module. Model to train
            dl_train: DataLoader. DataLoader containing the training data
            dl_val: DataLoader. DataLoader containting the validation data

        """
        tb = SummaryWriter()
        #images = next(iter(dl_train))
        #tb.add_graph(model, images[0])
        
        self.verbose = verbose
        
        optimizer = model.configure_optimizers()
        train_loss_epochs = []
        train_acc_epochs = []
        val_acc_epochs = []

        for e in range(self.nb_epochs):
            loss_train = []
            acc_train = []
            for batch_idx, batch in enumerate(dl_train):
                model.train()
                optimizer.zero_grad()
                loss, acc = model.training_step(batch, batch_idx)
                loss.backward()
                optimizer.step()
                loss_train.append(loss.item())
                acc_train.append(acc)

            loss_val = []
            acc_val = []
            if self.verbose:
                for batch_idx, batch in enumerate(dl_val):
                    model.eval()
                    with torch.no_grad():
                        loss, acc = model.validation_step(batch, batch_idx)
                        loss_val.append(loss.item())
                        acc_val.append(acc)
                avg_loss_train = round(sum(loss_train)/len(loss_train), 2)
                avg_acc_train = round(sum(acc_train)/len(acc_train), 2)
                train_loss_epochs.append(avg_loss_train)
                train_acc_epochs.append(avg_acc_train)

                avg_loss_val = round(sum(loss_val)/len(loss_val), 2)
                avg_acc_val = round(sum(acc_val)/len(acc_val), 2)
                val_acc_epochs.append(avg_acc_val)
                print(f'# Epoch {e+1}/{self.nb_epochs}:\t loss={avg_loss_train}\t loss_val={avg_loss_val}\t acc_val={avg_acc_val}')

                # Write to tensor board
                tb.add_scalar("Training loss", avg_loss_train, e)
                tb.add_scalar("Training loss", avg_loss_val, e)
                tb.add_scalar("Accuracy", avg_acc_val, e)

                for name, weight in model.named_parameters():
                    tb.add_histogram(name,weight, e)
                    tb.add_histogram(f'{name}.grad',weight.grad, e)

            tb.close()

        if self.verbose:
            return train_loss_epochs, train_acc_epochs, val_acc_epochs 

    def test(self, model, dl_test, test_verbose=True, return_acc=True):
        """ Test the model on the specified data 
        Args:
            model: Module. Model to train
            dl_test: DataLoader. DataLoader containting the test data
            test_verbose: bool. Wether the test result should be printed
        """

        loss_test = []
        acc_test = []
        for batch_idx, batch in enumerate(dl_test):
            model.eval()
            with torch.no_grad():
                loss, acc = model.test_step(batch, batch_idx)
                loss_test.append(loss.item())
                acc_test.append(acc)

        avg_loss_test = round(sum(loss_test)/len(loss_test), 2)
        avg_acc_test = round(sum(acc_test)/len(acc_test), 2)
        if test_verbose:
            print(f'loss_test={avg_loss_test}\t acc_test={avg_acc_test}')
        if return_acc: 
            return avg_acc_test
