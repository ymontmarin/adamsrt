import torch.nn as nn


def vgg16(num_classes):
    return VGG(
        cfg=[
            64, 64, 'M', 128, 128, 'M',
            256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512, 'M'
        ],
        dropout=False,
        small_images=True,
        bn_linear=False,
        num_classes=num_classes
    )


#########################
# Implementation of VGG #
#########################

class BasicBlock(nn.Module):
    def __init__(self, in_channels, x):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            x,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(x)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.relu(self.bn(self.conv(input)))

        return out


class VGG(nn.Module):
    def __init__(self, cfg, small_images, dropout, bn_linear, num_classes):
        super(VGG, self).__init__()
        self.cfg = cfg
        self.features = self._make_layers()
        if small_images:
            target_size = 512
        else:
            target_size = 4096
        if not dropout and not bn_linear:
            self.classifier = nn.Linear(target_size, num_classes)
        elif dropout and not bn_linear:
            self.classifier = nn.Sequential(
                nn.Linear(target_size, target_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(target_size, target_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(target_size, num_classes),
            )
        elif dropout and bn_linear:
            self.classifier = nn.Sequential(
                nn.Linear(target_size, target_size),
                nn.BatchNorm2d(target_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(target_size, target_size),
                nn.BatchNorm2d(target_size),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(target_size, num_classes),
            )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        for x in self.cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                block = BasicBlock(in_channels, x)
                layers += [block]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
