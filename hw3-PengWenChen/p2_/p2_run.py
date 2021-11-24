import argparse
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn as nn
import torch
import numpy as np
import pdb
from torch.optim import lr_scheduler
import torchvision
from PIL import Image
from torch.autograd import Variable
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from p2_.model import Generator, Discriminator
from p2_.utils import FaceDataset

class GAN():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.latent_dim = args.latent_dim

        self.beta1 = args.beta1
        self.nc = args.nc
        self.ngf = args.ngf
        self.ndf = args.ndf

        self.max_epoch = args.max_epoch
        self.restore = args.restore
        self.ckpt_dir = args.checkpoint_dir
        self.save_dir = args.save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_loader):
        fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
        real_label = 1.
        fake_label = 0.

        criterion = nn.BCELoss()
        netD = Discriminator(nc=self.nc, ndf=self.ndf).to(self.device)
        netG = Generator(latent_dim=self.latent_dim, nc=self.nc, ngf=self.ngf).to(self.device)
        optimizerD = torch.optim.Adam(netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999)) 

        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        for epoch in range(self.max_epoch):
            for batch_idx, img in enumerate(train_loader):
                img = img.to(self.device)
                ### Discriminator ###
                optimizerD.zero_grad()
                
                # real
                label = torch.full((self.batch_size,), real_label, dtype=torch.float, device=self.device)
                output = netD(img).view(-1)
                dloss_real = criterion(output, label)
                dloss_real.backward()
                D_x = output.mean().item()
                # fake
                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_img = netG(noise)
                label.fill_(fake_label)
                output = netD(fake_img.detach()).view(-1)
                dloss_fake = criterion(output, label)
                dloss_fake.backward()
                D_G_z1 = output.mean().item()
                dloss = dloss_real + dloss_fake
                optimizerD.step()

                ### Generator ###
                optimizerG.zero_grad()
                label.fill_(real_label)
                output = netD(fake_img).view(-1)
                gloss = criterion(output, label)
                gloss.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                if batch_idx % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.max_epoch, batch_idx, len(train_loader),
                            dloss.item(), gloss.item(), D_x, D_G_z1, D_G_z2))

                # Check how the generator is doing by saving G's output on fixed_noise
                
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                iters += 1

            for i, generate_batch in enumerate(img_list):
                torchvision.utils.save_image(generate_batch, os.path.join(self.save_dir, f'{i}.jpg') ,nrow=8)
            
            if epoch % 10 == 0 and epoch > 0:
                self.save_checkpoint(os.path.join(self.ckpt_dir,'%i.pth' % epoch), netD, netG, optimizerD, optimizerG)

    def save_checkpoint(self, checkpoint_path, netD, netG, optimizerD, optimizerG):
        print('\nStart saving ...')
        state = {'netD_state_dict': netD.state_dict(),
                'netG_state_dict': netG.state_dict(),
                'optimizerD' : optimizerD.state_dict(),
                'optimizerG' : optimizerG.state_dict()}
        torch.save(state, checkpoint_path)
        print('model saved to %s\n' % checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=False):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['netG_state_dict'])
        # if optimizer:
        #     optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s\n' % checkpoint_path)

    
    def test(self, checkpoint_path, output_dir):
        fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)

        netG = Generator(latent_dim=self.latent_dim, nc=self.nc, ngf=self.ngf).to(self.device)
        self.load_checkpoint(checkpoint_path, netG)

        fake = netG(fixed_noise).detach().cpu()
        img = vutils.make_grid(fake, padding=2, normalize=True)

        torchvision.utils.save_image(img, output_dir ,nrow=8)
        print('done')






