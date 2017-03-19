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
    upscale_factor = 4
    cuda = True
    batch_size = 16
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
        batch_size=batch_size,
        shuffle=True)

    validation_data_loader = DataLoader(
        dataset=val_set,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False)

    gennet_opts = dict()
    gennet_opts['upscale_factor'] = upscale_factor
    gennet = GenNet(net_opts=gennet_opts).train()

    start_epoch = 1
    content_criterion = nn.MSELoss()

    if cuda:
        gennet = gennet.cuda()
        content_criterion = content_criterion.cuda()

    optimizerG = Adam(gennet.parameters(), lr=0.0001)

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
            target = Variable(batch[0])
            input = Variable(batch[1])
            if cuda:
                target = target.cuda()
                input = input.cuda()
            
            # Train Generator
            optimizerG.zero_grad()
            reconstructed = gennet(input)

            content_loss = content_criterion(
                extract_y_channel(reconstructed, cuda),
                extract_y_channel(target, cuda))

            loss = content_loss
            loss.backward()
            optimizerG.step()

            epoch_loss_g += loss.data[0]
            total_psnr += psnr(loss.data[0])

            print("Epoch[{}]({}/{}): Loss(G): {:.4f}".format(
                epoch, i, len(training_data_loader), loss.data[0]))
        avg_loss_g = epoch_loss_g / len(training_data_loader)
        avg_psnr = total_psnr / len(training_data_loader)

        print("Epoch {} : Avg Loss(G): {:.4f}".format(
            epoch, avg_loss_g))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=1)
        plots.update([epoch, avg_loss_g], rind=1, cind=2, lind=1)
        plots.save()

    def validate(epoch):
        epoch_loss_g = 0
        total_psnr = 0
        for i, batch in enumerate(validation_data_loader, 1):
            target = Variable(batch[0])
            input = Variable(batch[1])
            if cuda:
                target = target.cuda()
                input = input.cuda()
            
            # Train Generator
            reconstructed = gennet(input)

            content_loss = content_criterion(
                extract_y_channel(reconstructed, cuda),
                extract_y_channel(target, cuda))

            loss = content_loss

            epoch_loss_g += loss.data[0]
            total_psnr += psnr(loss.data[0])

            print("Epoch[{}]({}/{}): Loss(G): {:.4f}".format(
                epoch, i, len(training_data_loader), loss.data[0]))
        avg_loss_g = epoch_loss_g / len(training_data_loader)
        avg_psnr = total_psnr / len(training_data_loader)

        print("Epoch {} : Avg Loss(G): {:.4f}".format(
            epoch, avg_loss_g))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=2)
        plots.update([epoch, avg_loss_g], rind=1, cind=2, lind=2)
        plots.save()

    def checkpoint(epoch):
        path = join(snapshot_dir, "gnet-epoch-{}-pretrain.pth".format(epoch))
        torch.save(gennet, path)

    for epoch in range(start_epoch, 200+1):
        train_epoch(epoch)
        validate(epoch)
        checkpoint(epoch)
    input("Finished: press enter key to close plots")


if __name__ == '__main__':
    pre_train()
