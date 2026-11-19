"""
cfg 記錄各層 CONV 與 BN 的輸出 channel 數
"""
import torch
import torch.nn.functional as F
from numpy import block
from torch import nn


# ----------------------
# Bottleneck Block
# ----------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        assert len(out_channels) == 3, (
            "Bottleneck requires out_channels list of length 3"
        )  # Hint: out_channels is lenth of three.

        ################################################
        # Please replace ??? with the correct variable #
        # example: in_channels, out_channels[0], ...   #
        ################################################
        self.conv1 = nn.Conv2d(
            in_channels, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.conv2 = nn.Conv2d(
            out_channels[0],
            out_channels[1],
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        self.conv3 = nn.Conv2d(
            out_channels[1],
            out_channels[2],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


# ----------------------
# ResNet
# ----------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, cfg, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.current_cfg_idx = 0

        # Conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.current_cfg_idx += 1
        self.inplanes = 64

        # Layer1~Layer4
        self.layer1 = self._make_layer(block, 64, layers[0], cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        #############################################################################
        # Figure out how to generate the correct layers and downsample based on cfg #
        #############################################################################
        layer = []

        for i in range(blocks):
            out_channels = cfg[self.current_cfg_idx : self.current_cfg_idx + 3]
            self.current_cfg_idx += 3

            if i == 0:
                layer.append(
                    block(
                        self.inplanes,
                        planes,
                        out_channels,
                        downsample=nn.Conv2d(  # In order to pruing, BN is dropped.
                            self.inplanes,
                            out_channels[2],
                            1,
                            stride=stride,
                            bias=False,
                        ),
                        stride=stride,
                    )
                )
            else:
                layer.append(block(self.inplanes, planes, out_channels))

            self.inplanes = out_channels[2]

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = (
            [64]
            + [64, 64, 256] * 3
            + [128, 128, 512] * 4
            + [256, 256, 1024] * 6
            + [512, 512, 2048] * 3
        )
    layers = [3, 4, 6, 3]
    return ResNet(
        Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels
    )


def ResNet101(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = (
            [64]
            + [64, 64, 256] * 3
            + [128, 128, 512] * 4
            + [256, 256, 1024] * 23
            + [512, 512, 2048] * 3
        )
    layers = [3, 4, 23, 3]
    return ResNet(
        Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels
    )


def ResNet152(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = (
            [64]
            + [64, 64, 256] * 3
            + [128, 128, 512] * 8
            + [256, 256, 1024] * 36
            + [512, 512, 2048] * 3
        )
    layers = [3, 8, 36, 3]
    return ResNet(
        Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels
    )


if __name__ == "__main__":
    # Sanity check: cfg fully consumed and feature map shape after each stage
    x = torch.randn(2, 3, 32, 32)
    expected = {
        "layer1": (2, 256, 16, 16),
        "layer2": (2, 512, 8, 8),
        "layer3": (2, 1024, 4, 4),
        "layer4": (2, 2048, 2, 2),
    }

    for name, (fn, layers) in {
        "ResNet50": (ResNet50, [3, 4, 6, 3]),
        "ResNet101": (ResNet101, [3, 4, 23, 3]),
        "ResNet152": (ResNet152, [3, 8, 36, 3]),
    }.items():
        model = fn().eval()
        print(f"===== {name} =====")

        cfg_len = 1 + sum(layers) * 3
        assert model.current_cfg_idx == cfg_len, (
            f"cfg idx {model.current_cfg_idx} != {cfg_len}"
        )

        with torch.no_grad():
            h = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            print(f"stem   {tuple(h.shape)}")
            for layer_name, shape in expected.items():
                h = getattr(model, layer_name)(h)
                print(f"{layer_name} {tuple(h.shape)}")
                assert tuple(h.shape) == shape, f"{layer_name}: expected {shape}"

            out = model(x)
            print(f"output {tuple(out.shape)}")
            assert tuple(out.shape) == (2, 10)

    print("All shape checks passed.")
