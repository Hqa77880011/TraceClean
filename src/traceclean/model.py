import torch
from torch import nn


class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            if stride != 1 or in_channels != out_channels else None
        )

    def forward(self, inputs):
        activated = torch.relu(self.bn1(inputs))
        shortcut = inputs if self.shortcut is None else self.shortcut(activated)
        output = self.conv1(activated)
        output = self.conv2(torch.relu(self.bn2(output)))
        return output + shortcut


class PreActResNet18(nn.Module):
    def __init__(self, num_classes, width=64):
        super().__init__()
        self.conv = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        layers = []
        in_channels = width
        for stage in range(4):
            out_channels = width * 2 ** stage
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(PreActBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        output = self.layers(self.conv(inputs))
        output = self.pool(torch.relu(self.bn(output))).flatten(1)
        return self.fc(output)
