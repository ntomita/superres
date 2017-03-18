import torch
from torch import nn
from torch import autograd
#from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from torchvision import models

class ResBlock(nn.Module):
    def __init__(self, n=64, s=1, f=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=n,
            out_channels=n,
            kernel_size=f,
            stride=s,
            padding=(f-1)//2)
        xavier_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(
            in_channels=n,
            out_channels=n,
            kernel_size=f,
            stride=s,
            padding=(f-1)//2)
        xavier_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(x)) + x
        return y


class DeconvBlock(nn.Module):
    def __init__(self,  n=64, f=3, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n,
            out_channels=n*upscale_factor ** 2,
            kernel_size=f,
            stride=1,
            padding=(f-1)//2)
        xavier_normal(self.conv.weight)
        self.pixsf = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixsf(self.conv(x))


class GenNet(nn.Module):

    def __init__(self, net_opts):
        # net_opts not used now
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        xavier_normal(self.conv1.weight)
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.res5 = ResBlock()
        self.res6 = ResBlock()
        self.res7 = ResBlock()
        self.res8 = ResBlock()
        self.res9 = ResBlock()
        self.res10 = ResBlock()
        self.res11 = ResBlock()
        self.res12 = ResBlock()
        self.res13 = ResBlock()
        self.res14 = ResBlock()
        self.res15 = ResBlock()
        self.deconv1 = DeconvBlock()
        self.deconv2 = DeconvBlock()
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)
        xavier_normal(self.conv2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.conv2(x)
        return x


class Skip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def vgg13_52():
    model = models.vgg13(pretrained=True)
    model.features = nn.Sequential(*list(model.features.children())[:-1])  # remove last max pooling
    model.classifier = Skip()
    return model

class DisNet(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
        