from torchvision.transforms import Compose, RandomCrop, ToTensor, Scale, RandomHorizontalFlip
from torch.utils.data import Dataset
from torch import cat, stack
from os.path import join
from os import listdir
from PIL import Image
from utils import is_image


def load_image(file_path):
    return Image.open(file_path)


def load_luminance(file_path):
    y, _, _ = load_image(file_path).convert('YCbCr').split()
    return y


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


def transform_target_batch(crop_size):
    def transform(image):
        patches = extract_subimages(image, crop_size, crop_size)
        patches = [ToTensor()(x) for x in patches]
        return stack(patches, 0)
    return transform


def transform_input_batch(crop_size, upscale_factor):
    def transform(image):
        patches = extract_subimages(image, crop_size, crop_size)
        patches = [Compose([Scale(crop_size//upscale_factor), ToTensor()])(x) for x in patches]
        return stack(patches, 0)
    return transform


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
        transform_target=transform_target_batch(crop_size),
        transform_input=transform_input_batch(crop_size, upscale_factor))


def get_val_set(opts):
    root_dir = opts['root_dir']
    test_dir = join(root_dir, 'val')
    upscale_factor = opts['upscale_factor']
    crop_size = opts['crop_size']
    crop_size = fit_crop_size(crop_size, upscale_factor)
    return ImageDataset(
        test_dir,
        transform_target=transform_target_batch(crop_size),
        transform_input=transform_input_batch(crop_size, upscale_factor))
    #This is wrong!!

def get_test_set(opts):
    root_dir = opts['root_dir']
    test_dir = join(root_dir, 'test')
    upscale_factor = opts['upscale_factor']
    crop_size = opts['crop_size']
    crop_size = fit_crop_size(crop_size, upscale_factor)
    return ImageDataset(
        test_dir,
        transform_target=transform_target(crop_size),
        transform_input=transform_input(crop_size, upscale_factor))


def extract_subimages(image, crop_size, stride):
    subimages = []
    for w in range((image.width-crop_size) // stride):
        for h in range((image.height-crop_size) // stride):
            # (left, upper, right, lower)
            bbox = (w*stride, h*stride, w*stride+crop_size, h*stride+crop_size)
            subimages.append(image.crop(bbox))
    return subimages


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform_target=None, transform_input=None):
        super().__init__()
        self.images = [join(root_dir, img) for img in listdir(root_dir) if is_image(img)]
        self.transform_target = transform_target
        self.transform_input = transform_input

    def __getitem__(self, index):
        target = load_luminance(self.images[index]).copy()

        # if self.transform_target is not None:
        #     target = self.transform_target(target)

        # input = target.copy() # need?
        # if self.transform_input is not None:
        #     input = self.transform_input(input)

        # target = ToTensor()(target)
        # input = ToTensor()(input)
        # return target, input
        input = target.copy()
        target = self.transform_target(target)
        input = self.transform_input(input)
        print(target.size())
        print(input.size())
        return target, input

    def __len__(self):
        return len(self.images)



