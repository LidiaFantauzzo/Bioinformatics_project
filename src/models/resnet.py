import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
import copy

class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1):
        super(ResidualBlock, self).__init__()

        need_downsample = stride != 1 or in_channels != channels[-1]

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False,dilation=dilation),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False,dilation=dilation),
            nn.BatchNorm2d(channels[1]),
        )

        if need_downsample:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, channels[1], 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(channels[1])
            )

    def forward(self, x):
        if hasattr(self, "down"):
            residual = self.down(x)
        else:
            residual = x
        x = self.convs(x) + residual

        return x

class ResNet18(nn.Module):

    def __init__(self, in_channels):
        super(ResNet18, self).__init__()

        #outputstride = 16 
        #dilatation = [1,1,1,2]

        self.mod1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('drop', nn.Dropout(0.3)),
            ('maxpool', nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        channels = (64, 64)
        self.mod2 = nn.Sequential(
            ResidualBlock(64, channels, stride=1, dilation=1),
            ResidualBlock(64, channels, stride=1, dilation=1)
        )
        channels = (128,128)
        self.mod3 = nn.Sequential(
            ResidualBlock(64, channels, stride=2, dilation=1),
            ResidualBlock(128, channels, stride=1, dilation=1)
        )
        channels = (256,256)
        self.mod4 = nn.Sequential(
            ResidualBlock(128, channels, stride=2, dilation=1),
            ResidualBlock(256, channels, stride=1, dilation=1)
        )
        channels = (512,512)
        self.mod5 = nn.Sequential(
            #difference with ResNet18 ResidualBlock(256,channels,stride=2, padding = 1) no dilat
            #                         ResidualBlok(512,channels, stride =1, padding = 1) no dilat
            ResidualBlock(256, channels, stride=1, dilation=2),
            ResidualBlock(512, channels, stride=1, dilation=2)
        )

        #delete from ResNet18  AdaptiveAvgPool2d(output_size=(1, 1))
        #                      Linear(in_features=512, out_features=1000, bias=True)

        self.out_channels = 512

    def forward(self, x):

        x = self.mod1(x)
        x = self.mod2(x)
        x = self.mod3(x)
        x = self.mod4(x)
        out = self.mod5(x)

        return out

    def pretrain(self):
        netSource = models.resnet18(pretrained=True)

        #self.mod1.conv1.weight.data = copy.deepcopy(netSource.conv1.weight.data)
        self.mod1.bn1.weight.data = copy.deepcopy(netSource.bn1.weight.data)
        self.mod1.bn1.bias.data = copy.deepcopy(netSource.bn1.bias.data)
        self.mod1.bn1.running_mean.data = copy.deepcopy(netSource.bn1.running_mean.data)
        self.mod1.bn1.running_var.data = copy.deepcopy(netSource.bn1.running_var.data)
        self.mod2 = copy.deepcopy(netSource.layer1)
        self.mod3 = copy.deepcopy(netSource.layer2)
        self.mod4 = copy.deepcopy(netSource.layer3)
    
