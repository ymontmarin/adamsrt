# Spherical Perspective on Learning with Batch Norm.
Anonymous Author(s) 

## Dependencies
To have the proper dependencies to reproduce the experiments performed in the paper you can install the librairies in `requirements.txt`.
The exact version of librairies that has been used are mentioned
To use the code properly you need python3 and CUDA10.
To install the dependencies you can do
```bash
pip install -r requirements.txt
```

## Code usage
You can try all the new methods proposed in the paper AdamS, AdamSRT and SGD-MRT. You can also use the benchmark methods Adam, AdamG, AdamW, SGD.
The implementation of AdamSRT follows the algorithm from Appendix E. Algo 1.
As in the paper, training can be done on the dataset CIFAR10, CIFAR100, SVHN and the architecture ResNet18, VGG16 and ResNet20 (only for CIFAR10).
For the training, the best parameters found in a previous grid search (cf Appendix E. Table 4) are used for the chosen setting. They are referenced in the file `best_hyper_parameters.py`

To launch the training you just need to call `training_example.py` with the proper options
```bash
python training_example.py --optimizer=adamsrt --model=resnet18 --dataloader=cifar100
```
Options are :
```python
dataloader in ['cifar10', 'cifar100', 'svhn']
model in ['resnet18', 'resnet20', 'vgg16']
optimizer in ['adams', 'adamsrt', 'adamw', 'adam', 'adamg', 'sgd', 'sgdmrt']
```
