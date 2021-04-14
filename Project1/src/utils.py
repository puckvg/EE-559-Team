from src.dlc_practical_prologue import generate_pair_sets
import torch
from torch.utils.data import DataLoader, TensorDataset

def _load_data(data_id):
    # Load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(nb=1000)

    if data_id == 'class':
        # Prepare dataloader with class variable (Used to test auxiliary networks)
        ds_train_class = torch.utils.data.TensorDataset(torch.cat((train_input[:, 0:1, :, :], train_input[:, 1:2, :, :]), 0), torch.cat((train_classes[:, 0], train_classes[:, 1]), 0))
        ds_test_class = torch.utils.data.TensorDataset(torch.cat((test_input[:, 0:1, :, :], test_input[:, 1:2, :, :]), 0), torch.cat((test_classes[:, 0], test_classes[:, 1]), 0))
        dl_train_class = torch.utils.data.DataLoader(ds_train_class, batch_size=32, shuffle=True, num_workers=4)
        dl_test_class = torch.utils.data.DataLoader(ds_test_class, batch_size=32, shuffle=False, num_workers=4)
        return dl_train_class, dl_test_class

    if data_id == 'target':
        # Prepare dataloader with target variable (Used for networks without auxiliary loss)
        ds_train_target = torch.utils.data.TensorDataset(train_input, train_target)
        ds_test_target = torch.utils.data.TensorDataset(test_input, test_target)
        dl_train_target = torch.utils.data.DataLoader(ds_train_target, batch_size=32, shuffle=True, num_workers=4)
        dl_test_target = torch.utils.data.DataLoader(ds_test_target, batch_size=32, shuffle=False, num_workers=4)
        return dl_train_target, dl_test_target

    if data_id == 'all':
        # Prepare dataloader with target and class variable (Used for networks with auxiliary loss)
        ds_train_all = torch.utils.data.TensorDataset(train_input, train_classes, train_target)
        ds_test_all = torch.utils.data.TensorDataset(test_input, test_classes, test_target)
        dl_train_all = torch.utils.data.DataLoader(ds_train_all, batch_size=32, shuffle=True, num_workers=4)
        dl_test_all = torch.utils.data.DataLoader(ds_test_all, batch_size=32, shuffle=False, num_workers=4)
        return dl_train_all, dl_test_all

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