import sys
from os.path import join, basename, exists
from os import makedirs, remove
import tarfile
import zipfile
from io import BytesIO
from six.moves.urllib.request import urlopen
from PIL import Image
from utils.utils import is_image, filename_wo_ext


def download_aplus(dest='dataset'):
    """ Download BSDS300 and extract images under train/test folders.
        Resulting folders are following:
        -[dest]-Aplus-images-train
                            -Set5-test
                            -Set14-test
    """

    def in_set5(file_path):
        return file_path.find('Set5') != -1

    def in_set14(file_path):
        return file_path.find('Set14') != -1

    def in_train(file_path):
        """ The 91 images
        """
        return file_path.find('Training') != -1 and file_path.find('CVPR08-SR') != -1

    url = "http://www.vision.ee.ethz.ch/~timofter/software/AplusCodes_SR.zip"
    output_dir = join(dest, 'Aplus', 'images')
    if not exists(output_dir):
        makedirs(output_dir)
        tmp_file = join(dest, basename(url))
        if not exists(tmp_file):
            response = urlopen(url)
            buf_size = 16 * 1024
            with open(tmp_file, 'wb') as f:
                while True:
                    buf = response.read(buf_size)
                    if not buf:
                        break
                    f.write(buf)
        with zipfile.ZipFile(tmp_file) as f:
            pass
            train_dir = join(output_dir, 'train')
            set5_dir = join(output_dir, 'Set5', 'test')
            set14_dir = join(output_dir, 'Set14', 'test')
            makedirs(train_dir)
            makedirs(set5_dir)
            makedirs(set14_dir)
            for item in f.infolist():
                if is_image(item.filename):
                    if in_train(item.filename):
                        image = Image.open(BytesIO(f.read(item)))
                        image.save(join(
                            train_dir,
                            filename_wo_ext(item.filename)+'.jpg'))
                    elif in_set5(item.filename):
                        image = Image.open(BytesIO(f.read(item)))
                        image.save(join(
                            set5_dir,
                            filename_wo_ext(item.filename)+'.jpg'))
                    elif in_set14(item.filename):
                        image = Image.open(BytesIO(f.read(item)))
                        image.save(join(
                            set14_dir,
                            filename_wo_ext(item.filename)+'.jpg'))
        remove(tmp_file)
    return output_dir


def download_bsds500(dest='dataset'):
    """Download BSDS500 and extract images under train/test folders.
    """
    def in_test(file_path):
        return file_path.find('test') != -1 and file_path.find('images') != -1

    def in_val(file_path):
        return file_path.find('val') != -1 and file_path.find('images') != -1

    def in_train(file_path):
        return file_path.find('train') != -1 and file_path.find('images') != -1

    url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    output_dir = join(dest, 'BSDS500', 'images')
    if not exists(output_dir):
        makedirs(output_dir)
        tmp_file = join(dest, basename(url))
        if not exists(tmp_file):
            response = urlopen(url)
            buf_size = 16 * 1024
            with open(tmp_file, 'wb') as f:
                while True:
                    buf = response.read(buf_size)
                    if not buf:
                        break
                    f.write(buf)
        with tarfile.open(tmp_file) as f:
            train_dir = join(output_dir, 'train')
            val_dir = join(output_dir, 'val')
            test_dir = join(output_dir, 'test')
            makedirs(train_dir)
            makedirs(val_dir)
            for item in f.getmembers():
                if is_image(item.name):
                    if in_train(item.name):
                        item.name = basename(item.name)
                        f.extract(item, train_dir)
                    elif in_val(item.name):
                        item.name = basename(item.name)
                        f.extract(item, val_dir)
                    elif in_test(item.name):
                        item.name = basename(item.name)
                        f.extract(item, test_dir)
        remove(tmp_file)
    return output_dir


def download_bsds300(dest='dataset'):
    """Download BSDS300 and extract images under train/test folders.
    """
    def in_val(file_path):
        return file_path.find('test') != -1

    def in_train(file_path):
        return file_path.find('train') != -1

    url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    output_dir = join(dest, 'BSDS300', 'images')
    if not exists(output_dir):
        makedirs(output_dir)
        tmp_file = join(dest, basename(url))
        if not exists(tmp_file):
            response = urlopen(url)
            buf_size = 16 * 1024
            with open(tmp_file, 'wb') as f:
                while True:
                    buf = response.read(buf_size)
                    if not buf:
                        break
                    f.write(buf)
        with tarfile.open(tmp_file) as f:
            train_dir = join(output_dir, 'train')
            val_dir = join(output_dir, 'val')
            makedirs(train_dir)
            makedirs(val_dir)
            for item in f.getmembers():
                if is_image(item.name):
                    if in_train(item.name):
                        item.name = basename(item.name)
                        f.extract(item, train_dir)
                    if in_val(item.name):
                        item.name = basename(item.name)
                        f.extract(item, val_dir)
        remove(tmp_file)
    return output_dir


if __name__ == '__main__':
    #download_bsds500()
    #download_aplus()
    pass
