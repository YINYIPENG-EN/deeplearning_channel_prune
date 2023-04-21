import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.act4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.act5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.act6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(1024, 2048, 3, 1, 1, bias=False)
        self.act7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(2048, 4096, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.act5(x)

        x = self.conv6(x)
        x = self.act6(x)

        x = self.conv7(x)
        x = self.act7(x)

        out = self.conv8(x)
        return out