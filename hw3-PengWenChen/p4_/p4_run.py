import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
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
import csv

from p4_.model import _netG, _netD, _netF, _netC
from p4_.utils import weights_init, lr_scheduler, exp_lr_scheduler

class GTA():
    def __init__(self, config):
        self.max_epoch = config.max_epoch
        self.lr = config.lr
        self.batch_size = config.batch_size

        self.nz = 512 # latent space z
        self.ngf = 64
        self.ndf = 64
        self.nclasses = 10

        self.beta1 = 0.8 # for Adam, default=0.5
        self.lrd = 1e-4 # lr decay, default=0.0002 , 1e-4
        self.alpha = 0.3 # 0.3

        self.mean = np.array([0.44, 0.44, 0.44])
        self.std = np.array([0.19, 0.19, 0.19])
        self.adv_weight = 0.1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Defining networks and optimizers
        self.netG = _netG(self.ndf, self.ngf, self.nz, self.nclasses)
        self.netD = _netD(self.ndf, self.nclasses)
        self.netF = _netF(self.ndf)
        self.netC = _netC(self.ndf, self.nclasses)

        self.netG.to(self.device)
        self.netD.to(self.device)
        self.netF.to(self.device)
        self.netC.to(self.device)

        # Weight initialization
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.netF.apply(weights_init)
        self.netC.apply(weights_init)

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        # Defining optimizers
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerF = torch.optim.Adam(self.netF.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerC = torch.optim.Adam(self.netC.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

        self.ckpt_dir = config.checkpoint_dir
    
    def train(self, source_loader, target_loader, eval_loader, test_loader):
        reallabel = torch.FloatTensor(self.batch_size).fill_(self.real_label_val).to(self.device)
        fakelabel = torch.FloatTensor(self.batch_size).fill_(self.fake_label_val).to(self.device)

        reallabelv = Variable(reallabel)
        fakelabelv = Variable(fakelabel)

        iteration, best_score = 0, 0
        for epoch in range(self.max_epoch):
            self.netG.train()    
            self.netF.train()    
            self.netC.train()    
            self.netD.train()

            errF_list, errC_list, errD_list, errG_list = [], [], [], []
            for i, (datas, datat) in enumerate(zip(source_loader, target_loader)):
                # print(i)
                src_inputs, src_labels = datas
                tgt_inputs, __ = datat  
                
                src_inputs_unnorm = (((src_inputs*self.std[0]) + self.mean[0]) - 0.5)*2

                # Creating one hot vector
                labels_onehot = np.zeros((self.batch_size, self.nclasses+1), dtype=np.float32)
                for num in range(self.batch_size):
                    labels_onehot[num, src_labels[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((self.batch_size, self.nclasses+1), dtype=np.float32)
                for num in range(self.batch_size):
                    labels_onehot[num, self.nclasses] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)

                src_inputs, src_labels = src_inputs.to(self.device), src_labels.to(self.device)
                src_inputs_unnorm = src_inputs_unnorm.to(self.device)
                tgt_inputs = tgt_inputs.to(self.device)
                src_labels_onehot = src_labels_onehot.to(self.device)
                tgt_labels_onehot = tgt_labels_onehot.to(self.device)
                
                # Wrapping in variable
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                src_inputs_unnormv = Variable(src_inputs_unnorm)
                tgt_inputsv = Variable(tgt_inputs)
                src_labels_onehotv = Variable(src_labels_onehot)
                tgt_labels_onehotv = Variable(tgt_labels_onehot)

                # Updating D network
                self.netD.zero_grad()
                src_emb = self.netF(src_inputsv) #256, 3, 32, 32
                # pdb.set_trace()
                #src_emb.shape 256*128
                src_emb_cat = torch.cat((src_labels_onehotv, src_emb), 1)
                src_gen = self.netG(src_emb_cat) #256, 3, 32, 32

                tgt_emb = self.netF(tgt_inputsv)
                tgt_emb_cat = torch.cat((tgt_labels_onehotv, tgt_emb),1)
                tgt_gen = self.netG(tgt_emb_cat)

                src_realoutputD_s, src_realoutputD_c = self.netD(src_inputs_unnormv)   
                errD_src_real_s = self.criterion_s(src_realoutputD_s, reallabelv) 
                errD_src_real_c = self.criterion_c(src_realoutputD_c, src_labelsv) 

                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errD_src_fake_s = self.criterion_s(src_fakeoutputD_s, fakelabelv)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)          
                errD_tgt_fake_s = self.criterion_s(tgt_fakeoutputD_s, fakelabelv)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tgt_fake_s
                errD.backward(retain_graph=True)    
                self.optimizerD.step()

                # Updating G network
                self.netG.zero_grad()       
                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errG_c = self.criterion_c(src_fakeoutputD_c, src_labelsv)
                errG_s = self.criterion_s(src_fakeoutputD_s, reallabelv)
                errG = errG_c + errG_s
                errG.backward(retain_graph=True)
                self.optimizerG.step()
                

                # Updating C network
                self.netC.zero_grad()
                outC = self.netC(src_emb)   
                errC = self.criterion_c(outC, src_labelsv)
                errC.backward(retain_graph=True)    
                self.optimizerC.step()

                
                # Updating F network
                self.netF.zero_grad()
                errF_fromC = self.criterion_c(outC, src_labelsv)        

                src_fakeoutputD_s, src_fakeoutputD_c = self.netD(src_gen)
                errF_src_fromD = self.criterion_c(src_fakeoutputD_c, src_labelsv)*(self.adv_weight)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c = self.netD(tgt_gen)
                errF_tgt_fromD = self.criterion_s(tgt_fakeoutputD_s, reallabelv)*(self.adv_weight*self.alpha)
                
                errF = errF_fromC + errF_src_fromD + errF_tgt_fromD
                errF.backward()
                self.optimizerF.step()

                if iteration % 30 == 0:
                    print('\r epoch: %d, [iter: %d], errF: %f, errG: %f, errD: %f, errC: %f' \
                    % (epoch, iteration + 1, errF.item(), errG.item(), errD.item(), errC.item()))
                iteration += 1

                if self.lrd:
                    self.optimizerD = exp_lr_scheduler(self.optimizerD, epoch, self.lr, self.lrd, iteration)    
                    self.optimizerF = exp_lr_scheduler(self.optimizerF, epoch, self.lr, self.lrd, iteration)
                    self.optimizerC = exp_lr_scheduler(self.optimizerC, epoch, self.lr, self.lrd, iteration)  

            test_score = self.validate(eval_loader, test_loader)

            if best_score < test_score:
                save_path = os.path.join(self.ckpt_dir, f'{epoch}.pth')
                self.save_checkpoint(save_path, self.netF, self.netC, self.netG, self.netD,
                        self.optimizerF, self.optimizerC, self.optimizerG, self.optimizerD, epoch)
                if best_score < test_score:
                    best_score = test_score

    def save_checkpoint(self, checkpoint_path, netF, netC, netG, netD, optF, optC, optG, optD, epoch):
        print('\nStart saving ...')
        state = {'F_state_dict': netF.state_dict(),
                'C_state_dict': netC.state_dict(),
                'G_state_dict': netG.state_dict(),
                'D_state_dict': netD.state_dict(),
                'optimizer_F': optF.state_dict(),
                'optimizer_C': optC.state_dict(),
                'optimizer_G': optG.state_dict(),
                'optimizer_D': optD.state_dict(),
                'eopch': epoch}
        torch.save(state, checkpoint_path)
        print('model saved to %s\n' % checkpoint_path)

    def validate(self, eval_loader, test_loader):
        # self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        # self.netD.eval()

        with torch.no_grad():
            eval_correct = []
            eval_loss_list = []
            for batch_idx, (eval_data, eval_label) in enumerate(eval_loader):
                    eval_data = eval_data.to(self.device)
                    eval_label = eval_label.to(self.device)
                    
                    class_logits = self.netC(self.netF(eval_data))
                    eval_loss_list.append(self.criterion_c(class_logits, eval_label).item())
                    eval_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == eval_label).item())

            print('Eval set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        np.mean(eval_loss_list), sum(eval_correct), len(eval_loader.dataset),
                        100. * sum(eval_correct) / len(eval_loader.dataset)))

        with torch.no_grad():
            test_correct = []
            test_loss_list = []
            for batch_idx, (test_data, test_label) in enumerate(test_loader):
                test_data = test_data.to(self.device)
                test_label = test_label.to(self.device)

                class_logits = self.netC(self.netF(test_data))
                test_loss_list.append(self.criterion_c(class_logits, test_label).item())
                test_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == test_label).item())

            print('Test set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    np.mean(test_loss_list), sum(test_correct), len(test_loader.dataset),
                    100. * sum(test_correct) / len(test_loader.dataset)))
        return 100. * sum(test_correct) / len(test_loader.dataset)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.netF.load_state_dict(state['F_state_dict'])
        self.netC.load_state_dict(state['C_state_dict'])
        print('model loaded from %s\n' % checkpoint_path)

    def test(self, test_loader, model_dir, output_dir):
        self.load_checkpoint(model_dir)
        self.netF.eval()
        self.netC.eval()

        file_name_list = []
        pred_list = []
        with torch.no_grad():
            for batch_idx, (file_name, test_data) in enumerate(test_loader):
                test_data = test_data.to(self.device)

                class_logits = self.netC(self.netF(test_data))
                file_name_list.append(file_name[0])
                pred_list.append(torch.argmax(class_logits, dim=1).item())

        csv_path = output_dir
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_name', 'label'])
            index=0
            for file_name in file_name_list:
                print(file_name, end='\r')
                try:
                    writer.writerow([file_name, pred_list[index]])
                except:
                    pdb.set_trace()
                index += 1

    def tsne(self, target_name, test_loader, eval_loader, model_dir, img1_dir, img2_dir):
        self.load_checkpoint(model_dir)
        self.netF.eval()

        feature_list = []
        label_list = []
        domain_list = []
        with torch.no_grad():
            for batch_idx, (test_data, test_label) in enumerate(test_loader):
                print(batch_idx, end='\r')
                if batch_idx % 10==0:
                    test_data = test_data.to(self.device)
                    
                    feature = self.netF(test_data)
                    feature = feature.squeeze().cpu().numpy()

                    feature_list.append(feature)
                    label_list.append(test_label)
                    domain_list.append(0)

            for batch_idx, (eval_data, eval_label) in enumerate(eval_loader):
                print(batch_idx, end='\r')
                # if batch_idx % 10==0:
                eval_data = eval_data.to(self.device)

                feature = self.netF(eval_data)
                feature = feature.squeeze().cpu().numpy()

                feature_list.append(feature)
                label_list.append(eval_label)
                domain_list.append(1)
        
        # pdb.set_trace()
        print('Start tsne')
        X_tsne = TSNE(n_components=2).fit_transform(feature_list)
        print('end tsne')
        
        plt.figure(figsize=(16,10))
        for i in [0,1,2,3,4,5,6,7,8,9]:
            if i==0:
                color="#FF0000"
                dig = "0"
            elif i==1:
                color="#800000"
                dig = "1"
            elif i==2:
                color="#FFFF00"
                dig = "2"
            elif i==3:
                color="#808000"
                dig = "3"
            elif i==4:
                color="#00FF00"
                dig = "4"
            elif i==5:
                color="#008000"
                dig = "5"
            elif i==6:
                color="#00FFFF"
                dig = "6"
            elif i==7:
                color="#008080"
                dig = "7"
            elif i==8:
                color="#0000FF"
                dig = "8"
            elif i==9:
                color="#800080"
                dig = "9"

            xy = X_tsne[np.array(label_list)==i]
            plt.scatter(xy[:,0], xy[:,1], c=color, label=dig,
                    alpha=0.8)
        plt.legend()
        plt.title("Digit")
        # plt.scatter(X_tsne[:,0], X_tsne[:,1], c=label_list,)
        plt.savefig(img1_dir)
        plt.close()

        plt.figure(figsize=(16,10))
        for i in [0,1]:
            if i==0:
                color = "#FF0000"
                dom = 'target'
            elif i==1:
                color="#0000FF"
                dom = "source"
            xy = X_tsne[np.array(domain_list)==i]
            plt.scatter(xy[:,0], xy[:,1], c=color, label=dom,
                    alpha=0.8)
        plt.legend()
        plt.title("Domain")
        # plt.scatter(X_tsne[:,0], X_tsne[:,1], c=domain_list,)
        plt.savefig(img2_dir)
        plt.close()
        pdb.set_trace()
