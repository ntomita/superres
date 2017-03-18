from os.path import splitext, basename
from math import log10

# def is_image(file_path):
#     return file_path.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp']


def is_image(file_path):
    _, ext = splitext(file_path)
    return ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']


def filename_wo_ext(file_path):
    filename, _ = splitext(basename(file_path))
    return filename


def psnr(mse):
    return 10*log10(1/mse)
