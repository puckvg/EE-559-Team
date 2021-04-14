import torch

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
        self.verbose = verbose
        
        optimizer = model.configure_optimizers()
        for e in range(self.nb_epochs):
            loss_train = []
            for batch_idx, batch in enumerate(dl_train):
                model.train()
                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx)
                loss.backward()
                optimizer.step()
                loss_train.append(loss.item())

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
                avg_loss_val = round(sum(loss_val)/len(loss_val), 2)
                avg_acc_val = round(sum(acc_val)/len(acc_val), 2)
                print(f'# Epoch {e+1}/{self.nb_epochs}:\t loss={avg_loss_train}\t loss_val={avg_loss_val}\t acc_val={avg_acc_val}')

    def test(self, model, dl_test):
        """ Test the model on the specified data 
        Args:
            model: Module. Model to train
            dl_test: DataLoader. DataLoader containting the test data
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
        print(f'loss_test={avg_loss_test}\t acc_test={avg_acc_test}')
