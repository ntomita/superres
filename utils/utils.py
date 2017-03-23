from os.path import splitext, basename
from math import log10
import torch
from torch import nn


# def is_image(file_path):
#     return file_path.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp']

def extract_y_channel(x, cuda=False):
    """RGB to Y (YCbCr)
        [T,C,H,W]->[T,1,H,W]
    """
    #x = (x*255).clamp(0, 255)  # [0,1]->[0,255]
    filters = torch.FloatTensor([0.299, 0.587, 0.114]).resize_(1, 3, 1, 1)
    filters = torch.autograd.Variable(filters)
    if cuda:
        filters = filters.cuda()
    y = nn.functional.conv2d(x, filters)  # + 16
    return y  # .clamp(0, 255)/255.0


def is_image(file_path):
    _, ext = splitext(file_path)
    return ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']


def filename_wo_ext(file_path):
    filename, _ = splitext(basename(file_path))
    return filename


def psnr(mse):
    return 10*log10(1/mse)
