# Spherical Perspective on Learning with Batch Norm. New methods : AdamSRT
[Simon Roburin](https://github.com/kdoublerotor)\*,
[Yann de Mont-Marin](https://github.com/ymontmarin)\*,
[Andrei Bursuc](XXXX),
[Renaud Marlet](XXXX),
[Patrick Pérez](XXXX),
[Mathieu Aubry](XXXX)
\
\*equal contribution
\
*nom_de_la_conf, 2020*

- [Project page](XXXXX)
- [Paper](XXXXX)

If you find this code useful for your research, consider citing our paper:
```
@INPROCEEDINGS{roburinmontmarin20_sphere_BN,
           title     = {Spherical Perspective on Learning with Batch Norm},
           author    = {Simon, Roburin and Yann, de Mont-Marin and Andrei, Bursuc and Renaud, Marlet and Patrick, Pérez and Mathieu, Aubry},
           booktitle = {XXXX},
           year      = {2020}
}
```


### Table of Content
- [Abstract](#abstract)
- [Setup](#setup)
- [New optimizers](#new-optimizers)
  - [Methods](#methods)
  - [Usage](#usage)
- [Benchmark](#benchmark)
  - [Results](#results)
  - [Train models with optimizer](#train-models-with-optimizer)
- [Acknowledgements](#acknowledgements)


## Abstract
Batch Normalization (BN) is a prominent deep learning technique. In spite of its apparent simplicity, its implications over optimization are yet to be fully understood. In this paper, we study the optimization of neural networks with BN layers from a geometric perspective. We leverage the radial invariance of groups of parameters, such as neurons for multi-layer perceptrons or filters for convolutional neural networks, and translate several popular optimization schemes on the L<sub>2</sub> unit hypersphere. This formulation and the associated geometric interpretation sheds new light on the training dynamics and the relation between different optimization schemes. In particular, we use it to derive the effective learning rate of Adam and stochastic gradient descent (SGD) with momentum, and we show that in the presence of BN layers, performing SGD alone is actually equivalent to a variant of Adam constrained to the unit hypersphere. Our analysis also leads us to introduce new variants of Adam. We empirically show, over a variety of datasets and architectures, that they improve accuracy in classification tasks.

This repository implements the new optimizer proposed in the paper AdamS and AdamSRT and gives a `train.py` script to reproduce the benchmark results presented in the paper.

## Setup
To use the package properly you need python3 and it is recommanded to use CUDA10 for acceleration. The Installation is as follow:

1. Clone the repo:
```bash
$ git clone https://github.com/ymontmarin/adamsrt
```

2. Install this repository and the dependencies using pip:
```bash
$ pip install -e adamsrt
```

With this, you can edit the code on the fly and import function and classes of the package in other project as well.

3. Optional. To uninstall this package, run:
```bash
$ pip uninstall adamsrt
```

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
        get_dataloader_cifar10, get_dataloader_cifar100, get_dataloader_SVHN
    optimizers.
        AdamSRT, AdamS, SGDMRT, AdamG
```


## New optimizers
### Methods
The paper introduce a geometrical framework which allows to identify behaviour of Adam which are not easily translated in the context of optimization on manifold. We introduce the rescaling and transport (RT) of the momentum and standardization (S) of the step division in Adam to neutralize these effects. We are able to apply these transformation to Adam with few lines of codes and introduce AdamS and AdamSRT.


### Usage
These optimizers (AdamSRT, AdamS) are conceived to give a particular treatment on layer followed by BN (or other normalization layer). 
To use  it with pytorch, you need to use paramgroups of pytorch (see the [doc](https://pytorch.org/docs/stable/optim.html#per-parameter-options)).
It allows you to specify the parameters followed by a normalization and activate the special treatment option `channel_wise=True` for these parameters.

The typical use for 2D convolutional networks where a convolutional layer is followed by a BN layer looks like:
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
Advanced details on the use of the optimizers can be found in `adamsrt/README.md`.

## Benchmark
### Results
Methods AdamSRT and Adam has been benchmark over a range of classification datasets and architecture. They are compared to the classical counterpart Adam and its variant (AdamW, AdamG) as well as state of the art SGD-M.

<sub><sup>
| Method          | CIFAR10 ResNet20 | CIFAR10 ResNet18 | CIFAR10 VGG16 | CIFAR100 ResNet18 | CIFAR100 VGG16 | SVHN ResNet18 | SVHN VGG16 |
| :-------------- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| SGD-M           | **92.39** (0.12) | **95.10** (0.04) | *93.56* (0.05) | **77.08** (0.18) | **73.77** (0.10) | **95.96** (0.15) | **95.95** (0.09) |
| Adam            | 90.98 (0.06) | 93.77 (0.20) | 92.83 (0.17) | 71.30 (0.36) | 68.43 (0.16) | 95.32 (0.23) | 95.57 (0.20) |
| AdamW           | 90.19 (0.24) | 93.61 (0.12) | 92.53 (0.25) | 67.39 (0.27) | 71.37 (0.22) | 95.38 (0.15) | 95.60 (0.08) |
| AdamG           | 91.64 (0.17) | 94.67 (0.12) | 93.41 (0.17) | 73.76 (0.34) | 70.17 (0.20) | 95.73 (0.05) | 95.70 (0.25) |
| AdamS (ours)   | 91.15 (0.11) | 93.95 (0.23) | 92.92 (0.11) | 74.44 (0.22) | 68.73 (0.27) | 95.75 (0.09) | 95.66 (0.09) |
| AdamSRT (ours) | *91.81* (0.20) | *94.92* (0.05) | **93.75** (0.06) | *75.28* (0.35) | *71.45* (0.13) | *95.84* (0.07) | *95.82* (0.05) |
</sub></sup>

Where AdamSRT outperform classic Adam and existing variations (AdamW and AdamG) and breaching the gap in performance with SGD-M even outperforming it on the task CIFAR10 with VGG16.

### Train models with optimizer
To reproduce the results of the paper, you can try all the proposed methods AdamS and AdamSRT. SGD-MRT can also be tested but leads to less systematic improvements.
You can use the benchmark methods Adam, AdamG, AdamW, SGD.
As in the paper, training can be done on public dataset CIFAR10, CIFAR100, SVHN and the architecture ResNet18, VGG16 and ResNet20 (only for CIFAR10).
For the training, the best parameters found in a previous grid search (cf Appendix E. Table 4) are used for the chosen setting. They are referenced in the file `best_hyper_parameters.py`.

The training is done over 400 epochs with a step-wise scheduling with 3 jumps.

To launch the training you just need to call `training.py` with the proper options
```bash
cd adamsrt
python training.py --optimizer=adamsrt --model=resnet18 --dataloader=cifar100
```
Logs will gives you the train and valid accuracy during the training as well as the test accuracy at the end of the training.

Options are :
```python
dataloader in ['cifar10', 'cifar100', 'svhn']
model in ['resnet18', 'resnet20', 'vgg16']
optimizer in ['adams', 'adamsrt', 'adamw', 'adam', 'adamg', 'sgd', 'sgdmrt']
```

## Acknowledgements
- For this project, we strongly relied on the [torch framework](https://github.com/pytorch/pytorch).

- All the experiments present in the paper and in the tabular were run on the [valeo.ai](https://github.com/valeoai) cluster gpu infrastructure

- We thank [Gabriel de Marmiesse](https://github.com/gabrieldemarmiesse) for his precious help to conduct successfully all of our experiments
