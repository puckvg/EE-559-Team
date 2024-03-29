import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.dlc_practical_prologue import generate_pair_sets


def normalize_input(train_input, test_input):
    mean = torch.mean(train_input)
    std = torch.std(train_input)
    train_input -= mean
    train_input /= std
    test_input -= mean
    test_input /= std
    return train_input, test_input


def _load_data(
    data_id, batch_size=32, split_lengths=[800, 200], nb_workers=0, normalize=True
):
    # Load data
    (
        train_input,
        train_target,
        train_classes,
        test_input,
        test_target,
        test_classes,
    ) = generate_pair_sets(nb=1000)
    if normalize:
        train_input, test_input = normalize_input(train_input, test_input)

    if data_id == "class":
        # Prepare dataloader with class variable (Used to test auxiliary networks)
        ds_train_class = TensorDataset(
            torch.cat((train_input[:, 0:1, :, :], train_input[:, 1:2, :, :]), 0),
            torch.cat((train_classes[:, 0], train_classes[:, 1]), 0),
        )

        # double lengths because splitting the two pictures doubles the size of the dataset
        split_lengths = [split_lengths[0] * 2, split_lengths[1] * 2]
        ds_train_class, ds_val_class = random_split(ds_train_class, split_lengths)
        ds_test_class = TensorDataset(
            torch.cat((test_input[:, 0:1, :, :], test_input[:, 1:2, :, :]), 0),
            torch.cat((test_classes[:, 0], test_classes[:, 1]), 0),
        )
        dl_train_class = DataLoader(
            ds_train_class, batch_size=batch_size, shuffle=True, num_workers=nb_workers
        )
        dl_val_class = DataLoader(
            ds_val_class, batch_size=batch_size, shuffle=False, num_workers=nb_workers
        )
        dl_test_class = DataLoader(
            ds_test_class, batch_size=batch_size, shuffle=False, num_workers=nb_workers
        )
        return dl_train_class, dl_val_class, dl_test_class

    if data_id == "target":
        # Prepare dataloader with target variable (Used for networks without auxiliary loss)
        ds_train_target = TensorDataset(train_input, train_target)
        ds_train_target, ds_val_target = random_split(ds_train_target, split_lengths)
        ds_test_target = TensorDataset(test_input, test_target)

        dl_train_target = DataLoader(
            ds_train_target, batch_size=batch_size, shuffle=True, num_workers=nb_workers
        )
        dl_val_target = DataLoader(
            ds_val_target, batch_size=batch_size, shuffle=False, num_workers=nb_workers
        )
        dl_test_target = DataLoader(
            ds_test_target, batch_size=batch_size, shuffle=False, num_workers=nb_workers
        )
        return dl_train_target, dl_val_target, dl_test_target

    if data_id == "all":
        # Prepare dataloader with target and class variable (Used for networks with auxiliary loss)
        ds_train_all = TensorDataset(train_input, train_classes, train_target)
        ds_train_all, ds_val_all = random_split(ds_train_all, split_lengths)
        ds_test_all = TensorDataset(test_input, test_classes, test_target)

        dl_train_all = DataLoader(
            ds_train_all, batch_size=batch_size, shuffle=True, num_workers=nb_workers
        )
        dl_val_all = DataLoader(
            ds_val_all, batch_size=batch_size, shuffle=False, num_workers=nb_workers
        )
        dl_test_all = DataLoader(
            ds_test_all, batch_size=batch_size, shuffle=False, num_workers=nb_workers
        )
        return dl_train_all, dl_val_all, dl_test_all


def load_class_data(normalize=True):
    """Loads a dataloader constisting of input images and the respective digit classification.
    DataLoader:
        x: train_input      N × 2 × 14 × 14     Images
        y: train_classes    N                   Classes of the two digits ∈ {0,...,9}
    """
    return _load_data("class", normalize=normalize)


def load_target_data(normalize=True):
    """Loads a dataloader constisting of input images and the respective is larger classification.
    DataLoader:
        x: train_input      N × 2 × 14 × 14     Images
        y: train_classes    N                   Class to predict ∈ {0, 1}
    """
    return _load_data("target", normalize=normalize)


def load_all_data(normalize=True):
    """Loads a dataloader constisting of input images and the respective is larger classification.
    DataLoader:
        x: train_input      N × 2 × 14 × 14     Images
        c: train_classes    N                   Classes of the two digits ∈ {0,...,9}
        t: train_target     N                   Class to predict ∈ {0, 1}
    """
    return _load_data("all", normalize=normalize)


def param_count(model):
    """Calculates the total number of parameters of the model.
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
    """Prints the total number of parameters of the model.
    Args:
        model: Module. Model of which we want to know the number or params
    """
    total, trainable = param_count(model)

    print(f"Total number of parameters:     {total}")
    print(f"Number of trainable parameters: {trainable}")
    return


def plot_training_epochs(
    nb_epochs,
    train_losses,
    train_accuracies,
    validation_accuracies,
    y_label_left="accuracy",
    y_label_right="cross entropy loss",
    savefig=None,
):
    """left plot train/val accuracy against epochs, right plot train loss
    against epochs"""
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_xlabel("epochs")
    axes[1].set_xlabel("epochs")
    axes[0].set_ylabel(y_label_left)
    axes[1].set_ylabel(y_label_right)
    axes[1].plot(list(range(nb_epochs)), train_losses, label="train", color="red")
    axes[0].plot(list(range(nb_epochs)), train_accuracies, label="train", color="red")
    axes[0].plot(
        list(range(nb_epochs)), validation_accuracies, label="validation", color="blue"
    )
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=300)
    return plt


def multi_plot_training_epochs(
    nb_epochs,
    train_losses,
    train_accuracies,
    validation_accuracies,
    labels,
    y_label_0="train accuracy",
    y_label_1="validation accuracy",
    y_label_2="train cross entropy loss",
    savefig=None,
):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    for i, ax in enumerate(axes):
        axes[i].set_xlabel("epochs")
    axes[0].set_ylabel(y_label_0)
    axes[1].set_ylabel(y_label_1)
    axes[2].set_ylabel(y_label_2)
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(train_losses) + 1)))

    for i, label in enumerate(labels):
        color = next(colors)
        axes[0].plot(
            list(range(nb_epochs)), train_accuracies[i], label=label, color=color
        )
        axes[1].plot(
            list(range(nb_epochs)), validation_accuracies[i], label=label, color=color
        )
        axes[2].plot(list(range(nb_epochs)), train_losses[i], label=label, color=color)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300)
    return plt


### m1 ###
def init_m1():
    m1 = LinearBaseline()
    return m1


### m2 ###
def init_m2():
    alpha = LinearAlpha()
    beta = LinearBeta(label_encoded=False)
    m2 = Siamese(
        alpha, beta, weight_aux=0.5, softmax=False, argmax=False, strategy="sum"
    )
    return m2


### m3 ###
def init_m3():
    le_net = LeNet()
    linear = nn.Linear(20, 2)
    m3 = Siamese(
        le_net,
        target=linear,
        weight_aux=0.0,
        strategy="sum",
        softmax=False,
        argmax=False,
    )
    return m3


### m4 ###
def init_m4():
    le_net = LeNet()
    tail_net = nn.Linear(20, 2)
    m4 = Siamese(
        le_net,
        softmax=False,
        argmax=False,
        strategy="sum",
        target=tail_net,
        weight_aux=0.8,
    )
    return m4


### m5 ###
def init_m5():
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m5 = Siamese(
        le_net,
        target=tail_net,
        weight_aux=0.6,
        softmax=False,
        argmax=False,
        strategy="sum",
    )
    return m5


### m6 ###
def init_m6():
    le_net = LeNet()
    m6 = Siamese(
        le_net, target=None, softmax=False, argmax=False, strategy="sum", weight_aux=0.0
    )
    return m6


### m7 ###
def init_m7():
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=False)
    m7 = Siamese(
        le_net,
        target=tail_net,
        weight_aux=0.6,
        strategy="sum",
        softmax=True,
        argmax=False,
    )
    return m7


### m8 ###
def init_m8():
    le_net = LeNet()
    tail_net = TailLinear(label_encoded=True)
    m8 = Siamese(
        le_net,
        target=tail_net,
        weight_aux=0.6,
        strategy="sum",
        softmax=False,
        argmax=True,
    )
    return m8


### m9 ###
def init_m9():
    tail = TailLinear(label_encoded=True)
    le_net = LeNet()
    m9 = Siamese(auxiliary=le_net, target=tail, softmax=False, strategy="sum")
    return m9


### m10 ###
def init_m10():
    tail = TailLinear(label_encoded=True)
    le_net = LeNet()
    m10 = Siamese(auxiliary=le_net, target=tail, softmax=False, strategy="random")
    return m10
