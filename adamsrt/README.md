# adamsrt
### Import and package tree
To import the package you just need:
```python
import adamsrt
```
The package contain pytorch `Optimizer` for the new optimizers (AdamS, AdamSRT) proposed in the paper as well as utilities to load classic models and dataset and other optimizer (pytorch implementation of AdamG and SGD variant SGD-MRT):
```
adamsrt.
    AdamSRT, AdamS
    models.
        resnet18, resnet20, vgg16
    dataloaders.
        get_dataloader_cifar10, get_dataloader_cifar100, get_dataloader_SVHN, get_dataloader_imagenet
    optimizers.
        AdamSRT, AdamS, SGDMRT, AdamG
```

### Usage of new optimizer
##### Basic usage
The new optimizers (AdamSRT, AdamS, SGDMRT) are conceived to give a particular treatment on layer followed by BN (or other normalization layer). 
To use  it with pytorch, you need to use paramgroups of pytorch (see the [doc](https://pytorch.org/docs/stable/optim.html#per-parameter-options)).
It allows you to specify the parameters followed by a normalization and activate the special treatment option `channel_wise=True` for these parameters.

The use looks like:
```python
from adamsrt import AdamSRT
par_groups = [{'params': model.conv_params(), 'channel_wise'=True},
              {'params': model.other_params()}]
optimizer = AdamSRT(par_groups, lr=0.001, betas=(0.9, 0.99), weight_decay=1e-4)
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
```
Typical implementation of methods to filter the param for standard network as proposed by [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html) are:
```python
class CustomModel(MyModel):
    ...

    def conv_params(self):
        conv_params = []
        for name, param in self.named_parameters():
            if any(key in name for key in {'conv', 'downsample.0'}):
                conv_params.append(param)
        return conv_params
```
##### Advanced usage
As explained above, the option `channel_wise=True` is suited for layer followed by a BN layer.

To generalized the use for a layer followed by any normalization layer (e.g. GroupNorm, LayerNorm...) we have the argument `channel_dims`.

You can specify  as the list of dims that distinct the entity that are normalized, meaning that normalization occurs on the other dimensions.

Typically in 2D convolutional neural network parameter tensor are of shape `CxHxWxC'`.
- BatchNorm normalize for each channel over previous channel and spatial dim : `channel_dims = [0]`.
- LayerNorm normalize all the layer and we have `channel_dims = []`.
- Individual rescaling (which is identical to classic optimization in the code) correspond to `channel_dims = [0,1,2,3]`.

And we have the following binding:
- `channel_dims=None` will throw the behavior of the classic optimizer.
- `channel_wise=True` is a binding for the most common case of a normalization in respect to channels (the first dimension): `channel_dims = [0]`.
