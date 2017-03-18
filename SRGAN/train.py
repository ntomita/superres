import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from os.path import join, exists, isfile, basename
from os import makedirs
import argparse
import re

from models import GenNet
from dataset import get_train_set, get_val_set

import sys
sys.path.append('..')
from data import download_bsds300
from utils.plot import Plots
from utils.utils import psnr

def train(argv=sys.argv[1:]):
    upscale_factor = 4
    cuda = False
    snapshot_dir = 'snapshot'
    if not exists(snapshot_dir):
        makedirs(snapshot_dir)
    train_opts = dict()
    train_opts['root_dir'] = download_bsds300(join('..', 'dataset'))
    train_opts['upscale_factor'] = upscale_factor
    train_opts['crop_size'] = 96
    train_set = get_train_set(train_opts)
    val_set = get_val_set(train_opts)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=16,
        shuffle=True)

    validation_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=16,
        shuffle=False)
    gennet_opts = dict()
    gennet_opts['upscale_factor'] = upscale_factor
    gennet = GenNet(net_opts=gennet_opts).train()
    start_epoch = 1
    criterion = nn.MSELoss()
    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    optimizer = Adam(gennet.parameters(), lr=0.0001)

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
            loss = criterion(gennet(input), target)
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
            mse = criterion(gennet(input), target)
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
        torch.save(gennet, path)

    for epoch in range(start_epoch, 10+1):
        train_epoch(epoch)
        validate(epoch)
        checkpoint(epoch)
    input("Finished: press enter key to close plots")


if __name__ == '__main__':
    train()