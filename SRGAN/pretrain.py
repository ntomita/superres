import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from os.path import join, exists, isfile, basename
from os import makedirs
import argparse
import re

from models import GenNet, DisNet, vgg13_52
from dataset import get_train_set, get_val_set

import sys
sys.path.append('..')
from data import download_bsds300
from utils.plot import Plots
from utils.utils import psnr, extract_y_channel


def pre_train(argv=sys.argv[1:]):
    """
    Available Arguments:
        batch_size     (b): batch size
            int
            Default: 16
        epochs         (n): number of epochs to run
            int
            Default: 5
        lr             (l): learning rate for Adam
            float
            Default: 0.0001 (1e-4)
        cuda           (c): use cuda
            true/false
            Default: false
        threads        (t): number of threads for data loader
            int
            Default: 4
        seed           (s): random seed
            int
            Default: 1234
        resume         (r): path to latest snapshot
            string
        dataset        (d): path to the root of dataset
            string         (should contain train/ folder)
            Defaut: bsds300
        val_iter       (v): number of iterations per epoch (use if dataset is large)
            int
            Default: Inf
        train_iter     (i): number of iterations per epoch (use if dataset is large)
            int
            Default: Inf
    example:
        python3 pretrain.py -n 200 -c -i 50000 -v 500 -d IMAGENET_FOLDER
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b',
                        type=int, default=16,
                        help='batch size (default:16)')
    parser.add_argument('--epochs', '-n',
                        type=int, default=5,
                        help='number of epochs to run')
    parser.add_argument('--lr', '-l',
                        type=float, default=1e-4,
                        help='learning rate for Adam (default:0.0001)')
    parser.add_argument('--cuda', '-c',
                        action='store_true',
                        help='use cuda')
    parser.add_argument('--threads', '-t',
                        type=int, default=4,
                        help='number of threads for data loader')
    parser.add_argument('--seed', '-s',
                        type=int, default=1234,
                        help='random seed')
    parser.add_argument('--resume', '-r',
                        type=str,
                        help='path to latest snapshot')
    parser.add_argument('--dataset', '-d',
                        type=str,
                        help='path to the root of dataset')
    parser.add_argument('--val_iter', '-v',
                        type=int, default=float('Inf'),
                        help='number of iterations per epoch')
    parser.add_argument('--train_iter', '-i',
                        type=int, default=float('Inf'),
                        help='number of iterations per epoch')
    args = parser.parse_args(argv)

    # FIXED VALUES (due to archs of discriminator)
    upscale_factor = 4
    crop_size = 96

    cuda = args.cuda
    batch_size = args.batch_size
    start_epoch = 1
    total_epochs = args.epochs

    snapshot_dir = 'snapshot'
    if not exists(snapshot_dir):
        makedirs(snapshot_dir)
    train_opts = dict()
    if args.dataset is not None:
        train_opts['root_dir'] = args.dataset
    else:
        train_opts['root_dir'] = download_bsds300(join('..', 'dataset'))
    train_opts['upscale_factor'] = upscale_factor
    train_opts['crop_size'] = crop_size
    train_set = get_train_set(train_opts)
    val_set = get_val_set(train_opts)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=args.threads,
        batch_size=batch_size,
        shuffle=True)

    validation_data_loader = DataLoader(
        dataset=val_set,
        num_workers=args.threads,
        batch_size=batch_size,
        shuffle=False)

    max_train_iter = min(len(training_data_loader), args.train_iter)
    max_val_iter = min(len(validation_data_loader), args.val_iter)

    gennet_opts = dict()
    gennet_opts['upscale_factor'] = upscale_factor
    gennet = GenNet(net_opts=gennet_opts).train()

    if args.resume is not None:
        if isfile(args.resume):
            gennet = torch.load(args.resume).train()
            start_epoch += int(re.findall("[+]?\d+[+]?", basename(args.resume))[0])

    content_criterion = nn.MSELoss()

    if cuda:
        gennet = gennet.cuda()
        content_criterion = content_criterion.cuda()

    optimizerG = Adam(gennet.parameters(), lr=args.lr)

    # Matlab like Plots
    plots = Plots(nrows=1, ncols=2)
    plots.add_line(rind=1, cind=1)
    plots.set_title("Average PSNR", rind=1, cind=1)
    plots.set_xlabel("epoch", rind=1, cind=1)
    plots.set_legend(['train', 'val'], rind=1, cind=1)

    plots.add_line(rind=1, cind=2)
    plots.set_title("Objective", rind=1, cind=2)
    plots.set_xlabel("epoch", rind=1, cind=2)
    plots.set_legend(['train_g', 'val_g'], rind=1, cind=2)

    def train_epoch(epoch):
        epoch_loss_g = 0
        total_psnr = 0
        for i, batch in enumerate(training_data_loader, 1):
            if i > max_train_iter:
                break
            target = Variable(batch[0])
            input = Variable(batch[1])
            if cuda:
                target = target.cuda()
                input = input.cuda()
            
            # Train Generator
            optimizerG.zero_grad()
            reconstructed = gennet(input)

            content_loss = content_criterion(reconstructed, target)

            loss = content_loss
            loss.backward()
            optimizerG.step()

            epoch_loss_g += loss.data[0]
            total_psnr += psnr(loss.data[0])

            print("Epoch[{}]({}/{}): Loss(G): {:.4f}".format(
                epoch, i, max_train_iter, loss.data[0]))
        avg_loss_g = epoch_loss_g / max_train_iter
        avg_psnr = total_psnr / max_train_iter

        print("Epoch {} : Avg Loss(G): {:.4f}".format(
            epoch, avg_loss_g))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=1)
        plots.update([epoch, avg_loss_g], rind=1, cind=2, lind=1)
        plots.save()

    def validate(epoch):
        epoch_loss_g = 0
        total_psnr = 0
        for i, batch in enumerate(validation_data_loader, 1):
            if i > max_val_iter:
                break
            target = Variable(batch[0])
            input = Variable(batch[1])
            if cuda:
                target = target.cuda()
                input = input.cuda()
            
            # Train Generator
            reconstructed = gennet(input)

            content_loss = content_criterion(reconstructed, target)

            loss = content_loss

            epoch_loss_g += loss.data[0]
            total_psnr += psnr(loss.data[0])

            print("Epoch[{}]({}/{}): Loss(G): {:.4f}".format(
                epoch, i, max_val_iter, loss.data[0]))
        avg_loss_g = epoch_loss_g / max_val_iter
        avg_psnr = total_psnr / max_val_iter

        print("Epoch {} : Avg Loss(G): {:.4f}".format(
            epoch, avg_loss_g))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=2)
        plots.update([epoch, avg_loss_g], rind=1, cind=2, lind=2)
        plots.save()

    def checkpoint(epoch):
        path = join(snapshot_dir, "gnet-epoch-{}-pretrain.pth".format(epoch))
        torch.save(gennet, path)

    for epoch in range(start_epoch, total_epochs+1):
        train_epoch(epoch)
        validate(epoch)
        save_model_every = 10
        if epoch % 10 == 0:
            checkpoint(epoch)
    input("Finished: press enter key to close plots")


if __name__ == '__main__':
    pre_train()
