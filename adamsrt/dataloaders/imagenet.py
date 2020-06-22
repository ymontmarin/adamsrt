from os.path import expanduser

from torch.utils.data import DataLoader
import torchvision.transforms as v_transforms
from torchvision.datasets import ImageNet

from adamsrt.config import IMAGENET_DATASET_ROOT_FOLDER, IMAGENET_NUM_WORKERS


# IMAGENET Mean and Std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_dataloader_imagenet(
    train_batch_size=256,
    test_batch_size=512,
    dataset_root_path=IMAGENET_DATASET_ROOT_FOLDER,
    num_workers=IMAGENET_NUM_WORKERS
):
    '''
    Function that build the Imagenet dataloader with classic transform
    Return a train_loader and valid_loader which are a random split of the
    train dataset according valid_split
    Canonical data augmentation is apply on the train_loader
    Normalization is done for for all loader
    The valid loader use the test_batch_size

    Arguments:
        train_batch_size (int): size of batch to use for train_loader
        test_batch_size (int): size to use for test_loader, valid_loader
        num_workers (int): nuber of worker to use
    '''
    dataset_root_path = expanduser(dataset_root_path)
    # Build regular data transformation
    train_transforms = v_transforms.Compose([
        v_transforms.RandomResizedCrop(224),
        v_transforms.RandomHorizontalFlip(),
        v_transforms.ToTensor(),
        v_transforms.Normalize(mean=MEAN, std=STD),
    ])

    test_transforms = v_transforms.Compose([
        v_transforms.Resize(256),
        v_transforms.CenterCrop(224),
        v_transforms.ToTensor(),
        v_transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_dataset = ImageNet(
        dataset_root_path,
        split='train',
        transform=train_transforms,
    )
    valid_dataset = ImageNet(
        dataset_root_path,
        split='val',
        transform=test_transforms,
        download=True
    )

    # We use valid split both as val and test loader
    data_loader_train = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    data_loader_valid = DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    return data_loader_train, data_loader_valid, data_loader_valid
