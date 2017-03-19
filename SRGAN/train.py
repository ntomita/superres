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
from utils.utils import psnr


def train(argv=sys.argv[1:]):
    upscale_factor = 4
    cuda = False
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
    vgg = vgg13_52()
    # for param in vgg.parameters():
    #     param.requires_grad = False
    disnet = DisNet().train()


    start_epoch = 1
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    label = Variable(torch.FloatTensor(batch_size))
    real_label = 1
    fake_label = 0

    if cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    optimizerG = Adam(gennet.parameters(), lr=0.0001)
    optimizerD = Adam(disnet.parameters(), lr=0.0001)

    plots = Plots(nrows=1, ncols=2)
    plots.add_line(rind=1, cind=1)
    plots.set_title("Average PSNR", rind=1, cind=1)
    plots.set_xlabel("epoch", rind=1, cind=1)
    plots.set_legend(['train', 'val'], rind=1, cind=1)

    plots.add_line(rind=1, cind=2)
    plots.add_line(rind=1, cind=2)
    plots.add_line(rind=1, cind=2)
    plots.set_title("Objective", rind=1, cind=2)
    plots.set_xlabel("epoch", rind=1, cind=2)
    plots.set_legend(['train_g', 'train_d', 'val_g', 'val_d'], rind=1, cind=2)

    def train_epoch(epoch):
        epoch_loss_g = 0
        epoch_loss_d = 0
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

            feature_r = vgg(reconstructed)
            feature_t = vgg(target)

            content_loss = content_criterion(feature_r, feature_t.detach())

            label.data.fill_(fake_label)
            adversarial_loss = adversarial_criterion(disnet(reconstructed), label)

            loss = content_loss + 1e-3*adversarial_loss
            loss.backward()
            optimizerG.step()

            # Train Discriminator
            optimizerD.zero_grad()
            reconstructed = gennet(input)

            fake_loss = adversarial_criterion(disnet(reconstructed.detach()), label)
            fake_loss.backward()

            label.data.fill_(real_label)
            real_loss = adversarial_criterion(disnet(target), label)
            real_loss.backward()
            optimizerD.step()

            epoch_loss_g += loss.data[0]
            epoch_loss_d += fake_loss.data[0] + real_loss.data[0]
            total_psnr += psnr(content_criterion(reconstructed, target).data[0])

            print("Epoch[{}]({}/{}): Loss(G): {:.4f} Loss(D): {:.4f}".format(
                epoch, i, len(training_data_loader), loss.data[0], fake_loss.data[0] + real_loss.data[0]))
        avg_loss_g = epoch_loss_g / len(training_data_loader)
        avg_loss_d = epoch_loss_d / len(training_data_loader)
        avg_psnr = total_psnr / len(training_data_loader)

        print("Epoch {} : Avg Loss(G): {:.4f} Avg Loss(D): {:.4f}".format(
            epoch, avg_loss_g, avg_loss_d))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=1)
        plots.update([epoch, avg_loss_g], rind=1, cind=2, lind=1)
        plots.update([epoch, avg_loss_d], rind=1, cind=2, lind=2)
        plots.save()

    def validate(epoch):
        epoch_loss_g = 0
        epoch_loss_d = 0
        total_psnr = 0
        for i, batch in enumerate(validation_data_loader, 1):
            target = Variable(batch[0])
            input = Variable(batch[1])
            if cuda:
                target = target.cuda()
                input = input.cuda()
            
            reconstructed = gennet(input)

            feature_r = vgg(reconstructed)
            feature_t = vgg(reconstructed)

            content_loss = content_criterion(feature_r, feature_t)

            label.data.fill_(fake_label)
            adversarial_loss = adversarial_criterion(disnet(reconstructed), label)

            loss = content_loss + 1e-3*adversarial_loss

            # Train Discriminator
            reconstructed = gennet(input)

            fake_loss = adversarial_criterion(disnet(reconstructed.detach()), label)

            label.data.fill_(real_label)
            real_loss = adversarial_criterion(disnet(target), label)

            epoch_loss_g += loss.data[0]
            epoch_loss_d += fake_loss.data[0] + real_loss.data[0]
            total_psnr += psnr(content_criterion(reconstructed, target).data[0])

            print("Epoch[{}]({}/{}): Loss(G): {:.4f} Loss(D): {:.4f}".format(
                epoch, i, len(training_data_loader), loss.data[0], fake_loss.data[0] + real_loss.data[0]))
        avg_loss_g = epoch_loss_g / len(training_data_loader)
        avg_loss_d = epoch_loss_d / len(training_data_loader)
        avg_psnr = total_psnr / len(training_data_loader)

        print("Epoch {} : Avg Loss(G): {:.4f} Avg Loss(D): {:.4f}".format(
            epoch, avg_loss_g, avg_loss_d))
        plots.update([epoch, avg_psnr], rind=1, cind=1, lind=2)
        plots.update([epoch, avg_loss_g], rind=1, cind=2, lind=3)
        plots.update([epoch, avg_loss_d], rind=1, cind=2, lind=4)
        plots.save()

    def checkpoint(epoch):
        path = join(snapshot_dir, "gnet-epoch-{}.pth".format(epoch))
        torch.save(gennet, path)
        path = join(snapshot_dir, "dnet-epoch-{}.pth".format(epoch))
        torch.save(disnet, path)


    for epoch in range(start_epoch, 10+1):
        train_epoch(epoch)
        validate(epoch)
        checkpoint(epoch)
    input("Finished: press enter key to close plots")


if __name__ == '__main__':
    train()