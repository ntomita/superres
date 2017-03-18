
from PIL import Image
import torch
from os.path import join, basename, dirname
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import numpy
import argparse
import sys


def combine(y_hr, cb_lr, cr_lr):
    cb_hr = cb_lr.resize(y_hr.size, Image.BICUBIC)
    cr_hr = cr_lr.resize(y_hr.size, Image.BICUBIC)
    return Image.merge('YCbCr', [y_hr, cb_hr, cr_hr]).convert('RGB')


def convert_to_y(net_output):
    y = net_output.data.squeeze().numpy()  # [1,1,h,w]->[h,w]
    y = (y*255.0).clip(0, 255)
    y = numpy.uint8(y)
    y = Image.fromarray(y, mode='L')  # L=Luminance, [h,w]->[w,h]
    return y


def test(argv=sys.argv[1:]):
    """
    Available Arguments:
        input  (i): path to input image (required)
            string
        output (o): path to save a final product
            string
            Default: current working directory
        model  (m): path to a model to be used (required)
            string
        cuda   (c): flag to use cuda
            true/false
            Default: false
    example:
        python3 test.py -i dataset/BSDS300/images/test/16077.jpg -m snapshot/net-epoch-30.pth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',
                        type=str, required=True,
                        help='path to input image')
    parser.add_argument('--output', '-o',
                        type=str,
                        help='path to save a final product')
    parser.add_argument('--model', '-m',
                        type=str, required=True,
                        help='path to a model to be used')
    parser.add_argument('--cuda', '-c',
                        action='store_true',
                        help='flag to use cuda')
    args = parser.parse_args(argv)

    if args.output is None:
        args.output = "sr_{}".format(basename(args.input))  # save in cwd
    y, cb, cr = Image.open(args.input).convert('YCbCr').split()
    width, height = y.size
    net = torch.load(args.model)
    y = ToTensor()(y)  # [w,h]->[1,h,w]
    input = Variable(y).view(1, -1, height, width)  # [w,h]->[1,1,h,w]

    if args.cuda:
        net = net.cuda()
        input = input.cuda()

    pred = net(input).cpu()

    y_hr = convert_to_y(pred)
    hr_image = combine(y_hr, cb, cr)
    hr_image.save(args.output)

if __name__ == '__main__':
    assert sys.version_info >= (3, 4)
    test()
