"""Models used in Project 2.

The original starter code only contained VGG-A without BatchNorm.  This file
keeps that baseline and adds the BatchNorm and configurable CIFAR models needed
for the project experiments.
"""
import numpy as np
import torch
from torch import nn

try:
    from utils.nn import init_weights_
except ImportError:  # Allows importing as a package from outside this folder.
    from ..utils.nn import init_weights_


def make_activation(name="relu", inplace=True):
    """Return an activation module by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if name == "elu":
        return nn.ELU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=inplace)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def conv_block(in_channels, out_channels, activation="relu", batch_norm=False, dropout=0.0):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(make_activation(activation))
    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))
    return layers

# ## Models implementation
def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n


class VGG_A(nn.Module):
    """VGG_A model

    size of Linear layers is smaller since input assumed to be 32x32x3, instead of
    224x224x3
    """

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGG_A_BatchNorm(nn.Module):
    """VGG-A baseline with BatchNorm after each convolution."""

    def __init__(self, inp_ch=3, num_classes=10, init_weights=True, activation="relu"):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            *conv_block(inp_ch, 64, activation=activation, batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            *conv_block(64, 128, activation=activation, batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            *conv_block(128, 256, activation=activation, batch_norm=True),
            *conv_block(256, 256, activation=activation, batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            *conv_block(256, 512, activation=activation, batch_norm=True),
            *conv_block(512, 512, activation=activation, batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 5
            *conv_block(512, 512, activation=activation, batch_norm=True),
            *conv_block(512, 512, activation=activation, batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            make_activation(activation),
            nn.Linear(512, 512),
            make_activation(activation),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


class VGG_A_Light(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # x = self.stage5(x)
        x = self.classifier(x.view(-1, 32 * 8 * 8))
        return x


class VGG_A_Dropout(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x


class ResidualBlock(nn.Module):
    """Small residual block for CIFAR-10 ablation experiments."""

    def __init__(self, in_channels, out_channels, stride=1, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = make_activation(activation)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + identity)


class CIFARConvNet(nn.Module):
    """Configurable CNN for the 60% CIFAR-10 part of the assignment."""

    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        channels=(64, 128, 256),
        activation="relu",
        batch_norm=True,
        dropout=0.1,
        classifier_hidden=512,
        residual=False,
        init_weights=True,
    ):
        super().__init__()
        if isinstance(channels, str):
            channels = tuple(int(item) for item in channels.split(",") if item)
        if len(channels) == 0:
            raise ValueError("channels must contain at least one value")

        layers = []
        in_channels = inp_ch
        for out_channels in channels:
            if residual:
                layers.append(ResidualBlock(in_channels, out_channels, activation=activation))
            else:
                layers.extend(conv_block(in_channels, out_channels, activation, batch_norm, dropout))
                layers.extend(conv_block(out_channels, out_channels, activation, batch_norm, 0.0))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, classifier_hidden),
            make_activation(activation),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_hidden, num_classes),
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


if __name__ == '__main__':
    print(get_number_of_parameters(VGG_A()))
    print(get_number_of_parameters(VGG_A_BatchNorm()))
    print(get_number_of_parameters(VGG_A_Light()))
    print(get_number_of_parameters(VGG_A_Dropout()))
    print(get_number_of_parameters(CIFARConvNet()))

    with torch.no_grad():
        x = torch.zeros(2, 3, 32, 32)
        print(VGG_A()(x).shape)
        print(VGG_A_BatchNorm()(x).shape)
        print(CIFARConvNet()(x).shape)
