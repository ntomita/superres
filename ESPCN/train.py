import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from os.path import join, exists, isfile, basename
from os import makedirs
import argparse
import re

from model import Net
from dataset import get_train_set, get_test_set, get_val_set

import sys
sys.path.append('..')
from data import download_bsds300
from utils.plot import Plots
from utils.utils import psnr


def train(argv=sys.argv[1:]):
    """
    Available Arguments:
        upscale_factor (f): upscale factor (required)
            int
        batch_size     (b): batch size
            int
            Default: 64
        epochs         (n): number of epochs to run
            int
            Default: 5
        lr             (l): learning rate for Adam
            float
            Default: 0.001
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

    example:
        python3 train.py -f 3 -n 30
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale_factor', '-f',
                        type=int, required=True,
                        help='upscale factor (required)')
    parser.add_argument('--batch_size', '-b',
                        type=int, default=64,
                        help='batch size (default:64)')
    parser.add_argument('--epochs', '-n',
                        type=int, default=5,
                        help='number of epochs to run')
    parser.add_argument('--lr', '-l',
                        type=float, default=0.0001,
                        help='learning rate for Adam (default:0.001)')
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
    args = parser.parse_args(argv)

    upscale_factor = args.upscale_factor
    cuda = args.cuda

    snapshot_dir = 'snapshot'
    if not exists(snapshot_dir):
        makedirs(snapshot_dir)

    train_opts = dict()
    train_opts['root_dir'] = download_bsds300(join('..', 'dataset'))
    train_opts['upscale_factor'] = upscale_factor
    train_opts['crop_size'] = 17 * upscale_factor
    train_set = get_train_set(train_opts)
    val_set = get_val_set(train_opts)

    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=args.threads,
        batch_size=args.batch_size,
        shuffle=True)

    validation_data_loader = DataLoader(
        dataset=val_set,
        num_workers=args.threads,
        batch_size=args.batch_size,
        shuffle=False)

    net_opts = dict()
    net_opts['upscale_factor'] = upscale_factor
    net = Net(net_opts=net_opts).train()

    start_epoch = 1
    if args.resume is not None:
        if isfile(args.resume):
            net = torch.load(args.resume).train()
            start_epoch += int(re.findall("[+]?\d+[+]?", basename(args.resume))[0])

    criterion = nn.MSELoss()
    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    optimizer = Adam(net.parameters(), lr=args.lr)

    plots = Plots(nrows=1, ncols=2)
    plots.add_line(rind=1, cind=1)
    plots.set_title("Average PSNR", rind=1, cind=1)
    plots.set_title("Objective", rind=1, cind=2)
    plots.set_xlabel("epoch", rind=1, cind=1)
    plots.set_xlabel("epoch", rind=1, cind=2)
    plots.add_line(rind=1, cind=2)
    plots.set_legend(['train', 'val'], rind=1, cind=1)
    plots.set_legend(['train', 'val'], rind=1, cind=2)

    def train_epoch(epoch):
        epoch_loss = 0
        total_psnr = 0
        for i, batch in enumerate(training_data_loader, 1):
            target = Variable(batch[0])
            input = Variable(batch[1])
            if cuda:
                target = target.cuda()
                input = input.cuda()
            optimizer.zero_grad()
            loss = criterion(net(input), target)
            epoch_loss += loss.data[0]
            total_psnr += psnr(loss.data[0])
            loss.backward()
            optimizer.step()
            print("Epoch[{}]({}/{}): Loss: {:.4f}".format(
                epoch, i, len(training_data_loader), loss.data[0]))
        avg_loss = epoch_loss / len(training_data_loader)
        avg_psnr = total_psnr / len(training_data_loader)
        print("Epoch {} : Avg Loss: {:.4f}".format(
            epoch, avg_loss))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=1)
        plots.update([epoch, avg_loss], rind=1, cind=2, lind=1)
        plots.save()

    def validate(epoch):
        total_loss = 0
        total_psnr = 0
        for batch in validation_data_loader:
            target = Variable(batch[0])
            input = Variable(batch[1])
            if cuda:
                target = target.cuda()
                input = input.cuda()
            mse = criterion(net(input), target)
            total_loss += mse.data[0]
            total_psnr += psnr(mse.data[0])
        avg_loss = total_loss / len(validation_data_loader)
        avg_psnr = total_psnr / len(validation_data_loader)
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=2)
        plots.update([epoch, avg_loss], rind=1, cind=2, lind=2)
        plots.save()

    def checkpoint(epoch):
        path = join(snapshot_dir, "net-epoch-{}.pth".format(epoch))
        torch.save(net, path)

    for epoch in range(start_epoch, args.epochs+1):
        train_epoch(epoch)
        validate(epoch)
        checkpoint(epoch)
    input("Finished: press enter key to close plots")

if __name__ == '__main__':
    train()
