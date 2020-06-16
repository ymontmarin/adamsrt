BEST_HYPER_PARAMETERS = {
    'cifar10': {
        'resnet18': {
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 1.5625e-05,
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 7.8125e-06,
            },
            'adamw': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 6.25e-05
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 7.8125e-06
            },
            'adamg': {
                'lr': 0.01,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            },
            'sgd': {
                'lr': 0.1,
                'weight_decay': 0.00025,
                'momentum': 0.9
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 0.00025,
                'momentum': 0.9
            }
        },
        'resnet20': {
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 3.125e-05,
                'momentum': 0.9
            },
            'sgd': {
                'lr': 0.1,
                'weight_decay': 0.00025,
                'momentum': 0.9
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 7.8125e-06
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            },
            'adamw': {
                'lr': 0.001,
                'betas': [0.9, 0.99]
            },
            'adamg': {
                'lr': 0.01,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            }
        },
        'vgg16': {
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 3.125e-05
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 6.25e-05,
                'momentum': 0.9
            },
            'sgd': {
                'lr': 0.1,
                'weight_decay': 0.00025,
                'momentum': 0.9
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 1.5625e-05
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 3.125e-05
            },
            'adamw': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 0.00025
            },
            'adamg': {
                'lr': 0.01,
                'betas': [0.9, 0.999],
                'weight_decay': 0.0005
            }
        }
    },
    'cifar100': {
        'resnet18': {
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 0.000125
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 0.0005,
                'momentum': 0.9
            },
            'sgd': {
                'lr': 0.1,
                'weight_decay': 0.0005,
                'momentum': 0.9
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 0.0
            },
            'adamw': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 0.000125
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 0.000125
            },
            'adamg': {
                'lr': 0.01,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            },
        },
        'vgg16': {
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 6.25e-05
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 0.000125,
                'momentum': 0.9
            },
            'sgd': {
                'lr': 0.1,
                'weight_decay': 0.0005,
                'momentum': 0.9
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 7.8125e-06
            },
            'adamw': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 1.5625e-05
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 6.25e-05
            },
            'adamg': {
                'lr': 0.01,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            }
        },
    },
    'svhn': {
        'resnet18': {
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 3.125e-05
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 0.0005,
                'momentum': 0.9
            },
            'sgd': {
                'lr': 0.1,
                'weight_decay': 0.0005,
                'momentum': 0.9
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 7.8125e-06
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 0.0
            },
            'adamw': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 7.8125e-06
            },
            'adamg': {
                'lr': 0.01,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            }
        },
        'vgg16': {
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 7.8125e-06
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 3.125e-05,
                'momentum': 0.9
            },
            'sgd': {
                'lr': 0.1,
                'weight_decay': 0.0005,
                'momentum': 0.9
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 0.00025
            },
            'adamw': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 3.125e-05
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0
            },
            'adamg': {
                'lr': 0.01,
                'betas': [0.9, 0.99],
                'weight_decay': 0.0005
            }
        }
    },
    'imagenet': {
        'resnet18': {
            'sgd': {
                'lr': 0.1,
                'weight_decay': 6.25e-05,
                'momentum': 0.9
            },
            'sgdmrt': {
                'lr': 0.1,
                'weight_decay': 6.25e-05,
                'momentum': 0.9
            },
            'adams': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 1.5625e-05
            },
            'adamsrt': {
                'lr': 0.001,
                'betas': [0.9, 0.999],
                'weight_decay': 7.8125e-06
            },
            'adam': {
                'lr': 0.001,
                'betas': [0.9, 0.99],
                'weight_decay': 7.8125e-06
            }
        }
    }
}
