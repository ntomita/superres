from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, RandomHorizontalFlip
from torch.utils.data import Dataset
from torch import cat, stack
from os.path import join
from os import listdir
from PIL import Image

import sys
sys.path.append('..')
from utils.utils import is_image


def load_image(file_path):
    return Image.open(file_path)


def transform_target(crop_size):
    """Ground truth image
    """
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(),
        ])


def transform_input(crop_size, upscale_factor):
    """LR of target image
    """
    return Compose([
        Scale(crop_size // upscale_factor),
        ])


def fit_crop_size(crop_size, upscale_factor):
    return crop_size - crop_size % upscale_factor


def get_train_set(opts):
    root_dir = opts['root_dir']
    train_dir = join(root_dir, 'train')
    upscale_factor = opts['upscale_factor']
    crop_size = opts['crop_size']
    crop_size = fit_crop_size(crop_size, upscale_factor)
    return ImageDataset(
        train_dir,
        transform_target=transform_target(crop_size),
        transform_input=transform_input(crop_size, upscale_factor))


def get_val_set(opts):
    root_dir = opts['root_dir']
    test_dir = join(root_dir, 'val')
    upscale_factor = opts['upscale_factor']
    crop_size = opts['crop_size']
    crop_size = fit_crop_size(crop_size, upscale_factor)
    return ImageDataset(
        test_dir,
        transform_target=transform_target(crop_size),
        transform_input=transform_input(crop_size, upscale_factor))


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform_target=None, transform_input=None):
        super().__init__()
        self.images = [join(root_dir, img) for img in listdir(root_dir) if is_image(img)]
        self.transform_target = transform_target
        self.transform_input = transform_input

    def __getitem__(self, index):
        target = load_image(self.images[index]).copy()
        if self.transform_target is not None:
            target = self.transform_target(target)

        input = target.copy() # need?
        if self.transform_input is not None:
            input = self.transform_input(input)

        target = ToTensor()(target)
        input = ToTensor()(input)
        return target, input

    def __len__(self):
        return len(self.images)

