import torch 


def normalize(tr_inp, te_inp):
    mean = torch.mean(tr_inp)
    std = torch.std(tr_inp)
    tr_inp -= mean 
    tr_inp /= mean 
    te_inp -= mean 
    te_inp /= mean 
    return tr_inp, te_inp 


def compute_class_accuracy(model, te_inp, te_target, mini_batch_size):
    incorrect = 0 

    for b in range(0, te_inp.size(0), mini_batch_size):
        pred = model(te_inp.narrow(0, b, mini_batch_size))
        label = torch.argmax(pred, dim=1)

        true_label = te_target.narrow(0, b, mini_batch_size)

        incorrect += torch.count_nonzero(true_label - label)

    acc = 1 - incorrect / te_target.size(0)
    return acc 


def train_model(model, criterion, tr_inp, tr_target, mini_batch_size, 
                optimizer, nb_epochs, verbose=False): 
    losses = torch.zeros(nb_epochs) 
    train_accuracy = torch.zeros(nb_epochs) 
    
    for i in range(nb_epochs):
        loss_epoch = 0 
        correct_epoch = 0 

        for b in range(0, tr_inp.size(0), mini_batch_size):
            tr_inp_chunk = tr_inp.narrow(0, b, mini_batch_size) 
            tr_target_chunk = tr_target.narrow(0, b, mini_batch_size) 
            output = model(tr_inp_chunk) 
            loss = criterion(output, tr_target_chunk)
            loss_epoch += loss.item()

            with torch.no_grad():
                pred = model(tr_inp_chunk)

            label = torch.argmax(pred, dim=1)
            true_label = tr_target_chunk 
            correct_epoch += (label == true_label).sum().item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy_i = correct_epoch / tr_target.size(0)
        train_accuracy[i] = train_accuracy_i
        losses[i] = loss_epoch 

        if verbose: 
            print('epoch', e) 
            print('loss', loss_epoch)
            print('train accuracy', train_accuracy_i)

    return model, losses, train_accuracy
