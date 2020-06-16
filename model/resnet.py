import torch
import torch.nn as nn


def resnet20(num_classes):
    return ResNet(
        num_classes=num_classes,
        init_in_planes=16,
        pool_size=8,
        block=BasicBlock,
        layer_num_blocks=[3, 3, 3],
    )


def resnet18(num_classes):
    return ResNet(
        num_classes=num_classes,
        init_in_planes=64,
        pool_size=4,
        block=BasicBlock,
        layer_num_blocks=[2, 2, 2, 2],
    )


#####################################
# ResNet code inspired from pytorch #
#####################################

def conv_k3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv_k1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_k3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_k3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                conv_k1(in_planes, planes, stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_k1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_k3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_k1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv_k1(in_planes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Implementation of ResNet.

    Args:
        - block: class, defined in custom_layers.resblock
        - num_blocks: list, of the size of each 3x3 conv in each layer
        - num_classes: int, number of classes
        - small_image: bool, if True we adapt the ResNet architecture to the
        one fo CIFAR10 from the original paper, else we choose ResNet
        architecture suited to ImageNet

    Inputs:
        - x: torch.Tensor of shape (B, C, W, H), B batch size, C nb of channels
        W, width of image, H, height of image

    Output:
        - y: torch.Tensor of shape (B, num_classes)
    """
    def __init__(
        self,
        block,
        layer_num_blocks,
        large=False,
        maxpool=False,
        num_classes=10,
        init_in_planes=16,
        adaptative_pool=False,
        pool_size=8,
        initializer=None
    ):
        super(ResNet, self).__init__()
        self.in_planes = init_in_planes

        # First conv
        if large:
            self.conv1 = nn.Conv2d(
                3, init_in_planes, kernel_size=7, stride=2,
                padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, init_in_planes, kernel_size=3, stride=1,
                padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(init_in_planes)
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        self.relu = nn.ReLU(inplace=True)

        # Block layers
        blocks_layers = []
        for i, num_blocks in enumerate(layer_num_blocks):
            planes = init_in_planes * (2**i)
            stride = 1 if i == 0 else 2
            blocks_layers.append(
                self._make_layer(block, planes, num_blocks, stride)
            )
        self.blocks_layers = nn.Sequential(*blocks_layers)

        # Output of the network
        if adaptative_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AvgPool2d(pool_size)
        self.linear = nn.Linear(planes * block.expansion, num_classes)

        # Make a custom initialization if needed
        if initializer is not None:
            initializer(self)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        # First block
        layers.append(
            block(self.in_planes, planes, stride)
        )
        self.in_planes = planes * block.expansion
        # Other blocks
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.blocks_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
