import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
import torch.optim as optim
import tqdm

# Imports from librairy
from adamsrt.models import resnet20, resnet18, vgg16

from adamsrt.dataloaders import (
    get_dataloader_cifar10,
    get_dataloader_cifar100,
    get_dataloader_SVHN
)
from adamsrt import AdamSRT, AdamS, SGDMRT
from adamsrt.optimizers import AdamG

# Imports best params from file in same place
from best_hyper_parameters import BEST_HYPER_PARAMETERS


##############
# Parameters #
##############

# Parameters fixed in the paper
N_EPOCH = 405
MILESTONES = [135, 225, 315]
GAMMA = 0.1
BEST_PATH = '/tmp/best_weights.pkl'


DATALOADERS = {
    'cifar10': {
        'dataloader_getter': get_dataloader_cifar10,
        'num_classes': 10,
    },
    'cifar100': {
        'dataloader_getter': get_dataloader_cifar100,
        'num_classes': 100,
    },
    'svhn': {
        'dataloader_getter': get_dataloader_SVHN,
        'num_classes': 10,
    },
}

MODELS = {
    'resnet18': resnet18,
    'resnet20': resnet20,
    'vgg16': vgg16,
}

OPTIMIZERS = {
    'adams': AdamS,
    'adamsrt': AdamSRT,
    'adamw': AdamW,
    'adam': Adam,
    'adamg': AdamG,
    'sgd': SGD,
    'sgdmrt': SGDMRT,
}


#################
# Actual Script #
#################

def main(dataloader_name, model_name, optimizer_name):
    # Print choices
    print(f'Doing optim with : {optimizer_name}')
    print(f'Using network : {model_name}')
    print(f'With dataset : {dataloader_name}')
    # Set up pytorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device - {device}')
    torch.manual_seed(0)

    # Get appropriate dataloader
    dataloader_getter = DATALOADERS[dataloader_name]['dataloader_getter']
    loader_train, loader_valid, loader_test = dataloader_getter()

    # Build model with appropriate classes
    model = MODELS[model_name](DATALOADERS[dataloader_name]['num_classes'])
    model.to(device)

    # Create losses
    loss = nn.CrossEntropyLoss()
    loss.to(device)

    # Create group_params
    if optimizer_name in {'adams', 'adamsrt', 'adamg', 'sgdmrt'}:
        # Prepare group params for special conv optimization
        # Split parameters in conv and other to activate channel optim
        conv_params, other_params = [], []
        for name, param in model.named_parameters():
            if any(key in name for key in {'conv', 'downsample.0'}):
                conv_params.append(param)
            else:
                other_params.append(param)
        # Just activate channel_wise for the conv
        conv_group = {'params': conv_params, 'channel_wise': True}
        other_group = {'params': other_params}
        # Get the group_params
        group_params = [conv_group, other_group]
    else:
        # All parameters are one group
        group_params = model.parameters()

    # Create optimizer with group_params and combo dataset/model/optimizer
    optimizer = OPTIMIZERS[optimizer_name](
        group_params,
        **BEST_HYPER_PARAMETERS[dataloader_name][model_name][optimizer_name]
    )

    # Add scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=MILESTONES,
        gamma=GAMMA
    )

    full_procedure(
        loader_train,
        loader_valid,
        loader_test,
        model,
        loss,
        optimizer,
        scheduler,
        device
    )


def full_procedure(
    loader_train,
    loader_valid,
    loader_test,
    model,
    loss,
    optimizer,
    scheduler,
    device
):
    # For each epoch make a train pass and a valid pass
    best_valid_acc = 0.
    i_best = 0
    for i in range(N_EPOCH):
        pass_on_data(
            loader_train,
            model,
            loss,
            device,
            optimizer=optimizer,
            keyword=f'Train epoch {i}'
        )
        valid_loss, valid_acc = pass_on_data(
            loader_valid,
            model,
            loss,
            device,
            keyword=f'Valid epoch {i}'
        )
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            i_best = i
            torch.save(model.state_dict(), BEST_PATH)
            print('SAVING BEST PARAMS')
        scheduler.step()
        print(f'lr set to {scheduler.get_lr()[0]}')
    # Make a final test pass on best params
    print(f'LOADING BEST PARAMS OF ITERATION {i_best}')
    checkpoint = torch.load(BEST_PATH)
    model.load_state_dict(checkpoint)
    pass_on_data(
        loader_test,
        model,
        loss,
        device,
        keyword=f'Test         '
    )


def pass_on_data(
    loader,
    model,
    loss,
    device,
    optimizer=None,
    keyword=''
):
    """
    Function that make a pass on datas
    If an optimizer is given it is a train pass
    """
    tqdm_batch = tqdm.tqdm(
        loader,
        desc="{} ".format(keyword),
        ascii=True
    )

    def loop():
        avg_loss = 0.
        avg_acc = 0.

        for batch_idx, (data, target) in enumerate(tqdm_batch):
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True)
            )
            # Predict
            pred = model(data)
            # Get loss
            cur_loss = loss(pred, target)
            # Backard
            if optimizer is not None:
                optimizer.zero_grad()
                cur_loss.backward()
                # Step
                optimizer.step()
            # Metric getting
            loss_value = cur_loss.item()
            if np.isnan(float(loss_value)):
                raise ValueError('Loss is nan during training...')
            _, pred = pred.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            n_correct = correct[:1].view(-1).float().sum(0).item()
            cur_acc = n_correct / target.size(0)
            # Update metrics
            avg_loss = (batch_idx * avg_loss + cur_loss) / (batch_idx + 1)
            avg_acc = (batch_idx * avg_acc + cur_acc) / (batch_idx + 1)
        tqdm_batch.close()
        return avg_loss, avg_acc

    if optimizer is not None:
        model.train()
        avg_loss, avg_acc = loop()
    else:
        model.eval()
        with torch.no_grad():
            avg_loss, avg_acc = loop()

    # Log loss and metrics
    print(f'{keyword} : loss={avg_loss} | acc={avg_acc}')
    return avg_loss, avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader", default="cifar100",
                        choices=list(DATALOADERS.keys()))
    parser.add_argument("--model", default="resnet18",
                        choices=list(MODELS.keys()))
    parser.add_argument("--optimizer", default="adamsrt",
                        choices=list(OPTIMIZERS.keys()))

    print(list(DATALOADERS.keys()))
    print(list(MODELS.keys()))
    print(list(OPTIMIZERS.keys()))

    args = parser.parse_args()
    renamed_args = {}
    for key, val in vars(args).items():
        renamed_args['_'.join([key, 'name'])] = val

    main(**renamed_args)
