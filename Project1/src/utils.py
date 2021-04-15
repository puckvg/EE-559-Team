from src.dlc_practical_prologue import generate_pair_sets
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def _load_data(data_id, batch_size=32, split_lengths=[800, 200], nb_workers=4):
    # Load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(nb=1000)

    if data_id == 'class':
        # Prepare dataloader with class variable (Used to test auxiliary networks)
        ds_train_class = TensorDataset(torch.cat((train_input[:, 0:1, :, :], train_input[:, 1:2, :, :]), 0), torch.cat((train_classes[:, 0], train_classes[:, 1]), 0))
        
        # double lengths because splitting the two pictures doubles the size of the dataset
        split_lengths = [split_lengths[0]*2, split_lengths[1]*2]
        ds_train_class, ds_val_class = random_split(ds_train_class, split_lengths)
        ds_test_class = TensorDataset(torch.cat((test_input[:, 0:1, :, :], test_input[:, 1:2, :, :]), 0), torch.cat((test_classes[:, 0], test_classes[:, 1]), 0))
        dl_train_class = DataLoader(ds_train_class, batch_size=batch_size, shuffle=True, num_workers=nb_workers)
        dl_val_class = DataLoader(ds_val_class, batch_size=batch_size, shuffle=False, num_workers=nb_workers)
        dl_test_class = DataLoader(ds_test_class, batch_size=batch_size, shuffle=False, num_workers=nb_workers)
        return dl_train_class, dl_val_class, dl_test_class

    if data_id == 'target':
        # Prepare dataloader with target variable (Used for networks without auxiliary loss)
        ds_train_target = TensorDataset(train_input, train_target)
        ds_train_target, ds_val_target = random_split(ds_train_target, split_lengths)
        ds_test_target = TensorDataset(test_input, test_target)
        
        dl_train_target = DataLoader(ds_train_target, batch_size=batch_size, shuffle=True, num_workers=nb_workers)
        dl_val_target = DataLoader(ds_val_target, batch_size=batch_size, shuffle=False, num_workers=nb_workers)
        dl_test_target = DataLoader(ds_test_target, batch_size=batch_size, shuffle=False, num_workers=nb_workers)
        return dl_train_target, dl_val_target, dl_test_target

    if data_id == 'all':
        # Prepare dataloader with target and class variable (Used for networks with auxiliary loss)
        ds_train_all = TensorDataset(train_input, train_classes, train_target)
        ds_train_all, ds_val_all = random_split(ds_train_all, split_lengths)
        ds_test_all = TensorDataset(test_input, test_classes, test_target)

        dl_train_all = DataLoader(ds_train_all, batch_size=batch_size, shuffle=True, num_workers=nb_workers)
        dl_val_all = DataLoader(ds_val_all, batch_size=batch_size, shuffle=False, num_workers=nb_workers)
        dl_test_all = DataLoader(ds_test_all, batch_size=batch_size, shuffle=False, num_workers=nb_workers)
        return dl_train_all, dl_val_all, dl_test_all

def load_class_data():
    """ Loads a dataloader constisting of input images and the respective digit classification.
    DataLoader:
        x: train_input      N × 2 × 14 × 14     Images
        y: train_classes    N                   Classes of the two digits ∈ {0,...,9}
    """
    return _load_data('class')

def load_target_data():
    """ Loads a dataloader constisting of input images and the respective is larger classification.
    DataLoader:
        x: train_input      N × 2 × 14 × 14     Images
        y: train_classes    N                   Class to predict ∈ {0, 1}
    """
    return _load_data('target')

def load_all_data():
    """ Loads a dataloader constisting of input images and the respective is larger classification.
    DataLoader:
        x: train_input      N × 2 × 14 × 14     Images
        c: train_classes    N                   Classes of the two digits ∈ {0,...,9}
        t: train_target     N                   Class to predict ∈ {0, 1}
    """
    return _load_data('all')


def param_count(model):
    """ Calculates the total number of parameters of the model.
    Args: 
        model: Module. Model of which we want to know the number or params
    
    Returns:
        total: Tensor. Total number of parameters
        trainable: Tensor. Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def print_param_count(model):
    """ Prints the total number of parameters of the model.
    Args: 
        model: Module. Model of which we want to know the number or params
    """
    total, trainable = param_count(model)
    
    print(f"Total number of parameters:     {total}")
    print(f"Number of trainable parameters: {trainable}")