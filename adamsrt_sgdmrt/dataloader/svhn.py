import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import SVHN
import torchvision.transforms as v_transforms

data_folder = "/tmp/SVHN"


def get_dataloader_SVHN(
    valid_split=0.05,
    train_batch_size=128,
    test_batch_size=1024,
    num_workers=1
):
    '''
    Function that build the SVHN dataloader with classic transform
    Return a train_loader and valid_loader which are a random split of the
    train dataset according valid_split
    Canonical data augmentation is apply on the train_loader
    Normalization is done for for all loader
    The valid loader use the test_batch_size

    Arguments:
        valid_split (float): percentage to cut off from train dataset for
            valid loader
        train_batch_size (int): size of batch to use for train_loader
        test_batch_size (int): size to use for test_loader, valid_loader
        num_workers (int): nuber of worker to use
    '''
    # Build regular data transformation
    train_transforms = v_transforms.Compose([
        v_transforms.ToTensor(),
    ])

    test_transforms = v_transforms.Compose([
        v_transforms.ToTensor(),
    ])

    train_dataset = SVHN(
        data_folder,
        split='train',
        transform=train_transforms,
        download=True
    )
    test_dataset = SVHN(
        data_folder,
        split='test',
        transform=test_transforms,
        download=True
    )
    # Cut a part of train for valid
    n = len(train_dataset)
    # has been seeded with sacred special value _seed
    indices = torch.randperm(n)
    n_cut = n - int(valid_split * n)
    valid_dataset = Subset(train_dataset, indices[n_cut:])
    train_dataset = Subset(train_dataset, indices[:n_cut])

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    data_loader_valid = DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    data_loader_test = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return data_loader_train, data_loader_valid, data_loader_test
