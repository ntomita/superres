import torch
from torch import nn
from torch.nn import init


class Net(nn.Module):
    def __init__(self, net_opts):
        super().__init__()

        upscale_factor = net_opts['upscale_factor']

        self.tanh = nn.Tanh()

        prev_filters = 1
        num_filters = 64
        kernel_size = 5
        padding = (kernel_size-1) // 2
        self.conv1 = nn.Conv2d(
            in_channels=prev_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)
        self._initialize_weights(self.conv1)

        prev_filters = num_filters
        num_filters = 32
        kernel_size = 3
        padding = (kernel_size-1) // 2
        self.conv2 = nn.Conv2d(
            in_channels=prev_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)
        self._initialize_weights(self.conv2)

        prev_filters = num_filters
        num_filters = upscale_factor ** 2
        kernel_size = 3
        padding = (kernel_size-1) // 2
        self.conv3 = nn.Conv2d(
            in_channels=prev_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)
        self._initialize_weights(self.conv3)

        self.pixsf = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.pixsf(self.conv3(x))
        return x

    def _initialize_weights(self, layer):
        init.xavier_normal(layer.weight)
