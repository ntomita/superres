import torch
from torch import nn
from torch import autograd
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
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        xavier_normal(self.conv2.weight)
        self.bn = nn.BatchNorm2d(64)
        self.deconv1 = DeconvBlock()
        self.deconv2 = DeconvBlock()
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 1)
        xavier_normal(self.conv3.weight)

    def forward(self, x):
        xs = self.relu(self.conv1(x))

        x = self.res1(xs)
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
        x = self.bn(self.conv2(x))
        x = x + xs
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.conv3(x)
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


netspec_opts = dict()
netspec_opts['input_channels'] = 3
netspec_opts['layer_type'] = ['conv', 'lrelu',
                              'conv', 'lrelu', 'bn',
                              'conv', 'lrelu', 'bn',
                              'conv', 'lrelu', 'bn',
                              'conv', 'lrelu', 'bn',
                              'conv', 'lrelu', 'bn',
                              'conv', 'lrelu', 'bn',
                              'conv', 'lrelu', 'bn']
netspec_opts['num_filters'] = [64, 0,
                               64, 0, 64,
                               128, 0, 128,
                               128, 0, 128,
                               256, 0, 256,
                               256, 0, 256,
                               512, 0, 512,
                               512, 0, 512]
netspec_opts['kernel_size'] = [3, 0,
                               3, 0, 0,
                               3, 0, 0,
                               3, 0, 0,
                               3, 0, 0,
                               3, 0, 0,
                               3, 0, 0,
                               3, 0, 0]
netspec_opts['stride'] = [1, 0,
                          2, 0, 0,
                          1, 0, 0,
                          2, 0, 0,
                          1, 0, 0,
                          2, 0, 0,
                          1, 0, 0,
                          2, 0, 0]


def make_layers(nopts):
    n = len(nopts['layer_type'])
    layers = []
    prev_filters = nopts['input_channels']
    for i in range(n):
        if nopts['layer_type'][i] == 'conv':
            curr_filters = nopts['num_filters'][i]
            layers.append(nn.Conv2d(
                prev_filters,
                curr_filters,
                nopts['kernel_size'][i],
                nopts['stride'][i],
                (nopts['kernel_size'][i]-1)//2))
            prev_filters = curr_filters
        elif nopts['layer_type'][i] == 'lrelu':
            layers.append(nn.LeakyReLU())
        elif nopts['layer_type'][i] == 'bn':
            curr_filters = nopts['num_filters'][i]
            layers.append(nn.BatchNorm2d(curr_filters))
            prev_filters = curr_filters
    return nn.Sequential(*layers)


class DisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = make_layers(netspec_opts)
        self.classifier = nn.Sequential(
            nn.Linear(6 * 6 * 512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid())
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                xavier_normal(module.weight)


