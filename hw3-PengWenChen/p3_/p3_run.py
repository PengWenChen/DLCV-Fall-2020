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
import csv

from p3_.model import FeatureExtractor, LabelPredictor, DomainClassifier, CNNModel

class DANN():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lamb = args.lamb
        self.max_epoch = args.max_epoch
        self.restore = args.restore
        self.ckpt_dir = args.checkpoint_dir
        self.lamb2 = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def save_checkpoint(self, checkpoint_path, fe, lp, opt_F, opt_L):
        print('\nStart saving ...')
        state = {'FE_state_dict': fe.state_dict(),
                'LP_state_dict': lp.state_dict(),
                'optimizer_F': opt_F.state_dict(),
                'optimizer_L': opt_L.state_dict(),}
        torch.save(state, checkpoint_path)
        print('model saved to %s\n' % checkpoint_path)

    def save_domain_dkpt(self, checkpoint_path, dc, opt_D):
        print('\nStart saving ...')
        state = {'DC_state_dict': dc.state_dict(),
                'optimizer_D': opt_D.state_dict(),}
        torch.save(state, checkpoint_path)
        print('model saved to %s\n' % checkpoint_path)
    
    def save_checkpoint_github(self, checkpoint_path, model, optimizer, epoch):
        print('\nStart saving ...')
        state = {'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,}
        torch.save(state, checkpoint_path)
        print('model saved to %s\n' % checkpoint_path)        
        
    def train_2(self, source_loader, target_loader, eval_loader, test_loader): # For training source and target
        feature_extractor = FeatureExtractor()
        label_predictor = LabelPredictor()
        domain_classifier = DomainClassifier()

        feature_extractor.to(self.device)
        label_predictor.to(self.device)
        domain_classifier.to(self.device)

        optimizer_F = torch.optim.Adam(feature_extractor.parameters(), lr=self.lr)
        optimizer_L = torch.optim.Adam(label_predictor.parameters(), lr=self.lr)
        optimizer_D = torch.optim.Adam(domain_classifier.parameters(), lr=self.lr)

        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCEWithLogitsLoss()
        
        iteration = 0
        for epoch in range(self.max_epoch):
            feature_extractor.train()
            label_predictor.train()
            domain_classifier.train()

            d_loss_list = []
            l_loss_list = []
            train_correct_list = []
            domain_correct_list = []
            for batch_idx, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)):
                source_data = source_data.to(self.device)
                source_label = source_label.to(self.device)
                target_data = target_data.to(self.device)

                mixed_data = torch.cat([source_data, target_data], dim=0)
                domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(self.device)
                domain_label[:source_data.shape[0]] = 1

                optimizer_F.zero_grad()
                optimizer_L.zero_grad()
                optimizer_D.zero_grad()
                
                # Train Domain Classifier
                feature = feature_extractor(mixed_data)
                domain_logits = domain_classifier(feature.detach())
                d_loss = domain_criterion(domain_logits, domain_label)
                d_loss_list.append(d_loss.item())
                d_loss.backward()
                optimizer_D.step()

                # pdb.set_trace()
                domain_correct_list.append(torch.sum(torch.argmax(domain_logits, dim=1).squeeze() == domain_label.squeeze()).item())

                # Train Feature Extractor and Label Classifier
                class_logits = label_predictor(feature[:source_data.shape[0]])
                domain_logits = domain_classifier(feature)

                # lamb = self.get_lambda(epoch, self.max_epoch)
                l_loss = class_criterion(class_logits, source_label) - self.lamb * domain_criterion(domain_logits, domain_label)
                l_loss_list.append(l_loss.item())
                l_loss.backward()
                optimizer_F.step()
                optimizer_L.step()

                train_correct_list.append(torch.sum(torch.argmax(class_logits, dim=1) == source_label).item())

                if iteration % 30 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tDomain Loss: {:.8f}\tLambda: {:.8f}'.format(
                        epoch, batch_idx * len(source_data), len(source_loader.dataset),
                        100. * batch_idx / len(source_loader), np.mean(l_loss_list), np.mean(d_loss_list), self.lamb))
                iteration += 1

            if self.lamb2 <0.3:
                self.lamb +=0.001
            self.lamb2 -=0.1
            print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(
                sum(train_correct_list), len(source_loader.dataset),
                100. * sum(train_correct_list) / len(source_loader.dataset)))

            print('Domain Accuracy: {}/{} ({:.0f}%)\n'.format(
                sum(domain_correct_list), len(source_loader.dataset)+len(target_loader.dataset),
                100. * sum(domain_correct_list) / (len(source_loader.dataset)+len(target_loader.dataset))))
            
            feature_extractor.eval()
            label_predictor.eval()
            domain_classifier.eval()
            with torch.no_grad():
                eval_correct = []
                eval_loss_list = []
                for batch_idx, (eval_data, eval_label) in enumerate(eval_loader):
                    eval_data = eval_data.to(self.device)
                    eval_label = eval_label.to(self.device)

                    feature = feature_extractor(eval_data)
                    class_logits = label_predictor(feature)
                    eval_loss_list.append(class_criterion(class_logits, eval_label).item())
                    eval_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == eval_label).item())

                print('Eval set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        np.mean(eval_loss_list), sum(eval_correct), len(eval_loader.dataset),
                        100. * sum(eval_correct) / len(eval_loader.dataset)))

                test_correct = []
                test_loss_list = []
                for batch_idx, (test_data, test_label) in enumerate(test_loader):
                    test_data = test_data.to(self.device)
                    test_label = test_label.to(self.device)

                    feature = feature_extractor(test_data)
                    class_logits = label_predictor(feature)
                    test_loss_list.append(class_criterion(class_logits, test_label).item())
                    test_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == test_label).item())

                print('Test set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                        np.mean(test_loss_list), sum(test_correct), len(test_loader.dataset),
                        100. * sum(test_correct) / len(test_loader.dataset)))

            if epoch % 10 == 0 and epoch > 0:
                save_path = os.path.join(self.ckpt_dir, f'{epoch}.pth')
                save_path_D = os.path.join(self.ckpt_dir, f'{epoch}_D.pth')
                self.save_checkpoint(save_path, feature_extractor, label_predictor, optimizer_F, optimizer_L)
                self.save_domain_dkpt(save_path_D, domain_classifier, optimizer_D)       
    
    def get_lambda(self, epoch, max_epoch):
        p = epoch / max_epoch
        return 2. / (1+np.exp(-10.*p)) - 1. 

    def train_2_github(self, source_loader, target_loader, eval_loader, test_loader):
        model = CNNModel()
        model.to(self.device)
        
        # pdb.set_trace()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_class = torch.nn.NLLLoss()
        loss_domain = torch.nn.NLLLoss()

        for epoch in range(self.max_epoch):
            model.train()
            len_dataloader = min(len(source_loader), len(target_loader))
            data_source_iter = iter(source_loader)
            data_target_iter = iter(target_loader)

            loss_domain_list = []
            loass_class_list = []
            train_correct_list = []
            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / self.max_epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                # print(alpha)
                # training model using source data
                s_img, s_label = data_source_iter.next()
                s_img, s_label = s_img.to(self.device), s_label.to(self.device)

                model.zero_grad()
                batch_size = len(s_label)
                domain_label = torch.zeros(batch_size).long().to(self.device)
                class_output, domain_output = model(input_data=s_img, alpha=alpha)
                
                err_s_label = loss_class(class_output, s_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                data_target = data_target_iter.next()
                t_img, _ = data_target
                t_img = t_img.to(self.device)

                batch_size = len(t_img)

                domain_label = torch.ones(batch_size).long().to(self.device)
                _, domain_output = model(input_data=t_img, alpha=alpha)
                err_t_domain = loss_domain(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()

                loss_domain_list.append(err_t_domain.item() + err_s_domain.item())
                loass_class_list.append(err_s_label.item())
                train_correct_list.append(torch.sum(torch.argmax(class_output, dim=1) == s_label).item())

                if i % 30 == 0:
                    print('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                    % (epoch, i + 1, len_dataloader, err_s_label.item(), err_s_domain.item(), err_t_domain.item()))
            
            print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(
                sum(train_correct_list), len(source_loader.dataset),
                100. * sum(train_correct_list) / len(source_loader.dataset)))
            
            model.eval()
            with torch.no_grad():
                eval_correct = []
                eval_loss_list = []
                for batch_idx, (eval_data, eval_label) in enumerate(eval_loader):
                    eval_data = eval_data.to(self.device)
                    eval_label = eval_label.to(self.device)

                    # feature = feature_extractor(eval_data)
                    # class_logits = label_predictor(feature)
                    class_logits, _ = model(input_data=eval_data, alpha=alpha)
                    eval_loss_list.append(loss_class(class_logits, eval_label).item())
                    eval_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == eval_label).item())

                print('Eval set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        np.mean(eval_loss_list), sum(eval_correct), len(eval_loader.dataset),
                        100. * sum(eval_correct) / len(eval_loader.dataset)))

                test_correct = []
                test_loss_list = []
                for batch_idx, (test_data, test_label) in enumerate(test_loader):
                    test_data = test_data.to(self.device)
                    test_label = test_label.to(self.device)

                    # feature = feature_extractor(test_data)
                    # class_logits = label_predictor(feature)
                    class_logits, _ = model(input_data=test_data, alpha=alpha)
                    test_loss_list.append(loss_class(class_logits, test_label).item())
                    test_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == test_label).item())

                print('Test set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                        np.mean(test_loss_list), sum(test_correct), len(test_loader.dataset),
                        100. * sum(test_correct) / len(test_loader.dataset)))
            
            if epoch % 5 == 0 and epoch > 0:
                save_path = os.path.join(self.ckpt_dir, f'{epoch}.pth')
                self.save_checkpoint_github(save_path, model, optimizer, epoch)

    def load_checkpoint(self, checkpoint_path, model, optimizer=False):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['state_dict'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s\n' % checkpoint_path)
    
    def test(self, test_loader, model_dir, output_dir):
        model = CNNModel()
        model.to(self.device)

        self.load_checkpoint(model_dir, model)
        model.eval()

        file_name_list = []
        pred_list = []
        with torch.no_grad():
            test_correct = []
            for batch_idx, (file_name, test_data) in enumerate(test_loader):
                print(file_name[0], end='\r')
                test_data = test_data.to(self.device)

                class_logits, _ = model(input_data=test_data, alpha=1)
                file_name_list.append(file_name[0])
                pred_list.append(torch.argmax(class_logits, dim=1).item())
            # print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            #         sum(test_correct), len(test_loader.dataset),
            #         100. * sum(test_correct) / len(test_loader.dataset)))

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
        model = CNNModel()
        model.to(self.device)
        # pdb.set_trace()
        model = model.feature

        feature_list = []
        label_list = []
        domain_list = []
        with torch.no_grad():
            for batch_idx, (test_data, test_label) in enumerate(test_loader):
                print(batch_idx, end='\r')
                # if batch_idx % 10==0:
                test_data = test_data.to(self.device)
                test_data = test_data.expand(test_data.data.shape[0], 3, 28, 28)
                feature = model(test_data)
                feature = feature.view(-1, 50 * 4 * 4).squeeze().cpu().numpy()
                # pdb.set_trace()
                feature_list.append(feature)
                label_list.append(test_label)
                domain_list.append(0)
            for batch_idx, (eval_data, eval_label) in enumerate(eval_loader):
                print(batch_idx, end='\r')
                if batch_idx % 10==0:
                    eval_data = eval_data.to(self.device)
                    eval_data = eval_data.expand(eval_data.data.shape[0], 3, 28, 28)
                    feature = model(eval_data)
                    feature = feature.view(-1, 50 * 4 * 4).squeeze().cpu().numpy()

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

    def lower(self, source_loader, target_loader, eval_loader, test_loader):
        model = CNNModel()
        model.to(self.device)
        
        # pdb.set_trace()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_class = torch.nn.NLLLoss()
        loss_domain = torch.nn.NLLLoss()

        for epoch in range(self.max_epoch):
            model.train()
            len_dataloader = len(source_loader)
            data_source_iter = iter(source_loader)
            # data_target_iter = iter(target_loader)

            loss_domain_list = []
            loass_class_list = []
            train_correct_list = []
            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / self.max_epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                s_img, s_label = data_source_iter.next()
                s_img, s_label = s_img.to(self.device), s_label.to(self.device)

                model.zero_grad()
                batch_size = len(s_label)
                domain_label = torch.zeros(batch_size).long().to(self.device)
                class_output, domain_output = model(input_data=s_img, alpha=alpha)
                
                err_s_label = loss_class(class_output, s_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                # data_target = data_target_iter.next()
                # t_img, _ = data_target
                # t_img = t_img.to(self.device)

                # batch_size = len(t_img)

                # domain_label = torch.ones(batch_size).long().to(self.device)
                # _, domain_output = model(input_data=t_img, alpha=alpha)
                # err_t_domain = loss_domain(domain_output, domain_label)
                # err = err_t_domain + err_s_domain + err_s_label
                err = err_s_domain + err_s_label
                err.backward()
                optimizer.step()

                # loss_domain_list.append(err_t_domain.item() + err_s_domain.item())
                loss_domain_list.append(err_s_domain.item())
                loass_class_list.append(err_s_label.item())
                train_correct_list.append(torch.sum(torch.argmax(class_output, dim=1) == s_label).item())

                if i % 30 == 0:
                    print('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f' \
                    % (epoch, i + 1, len_dataloader, err_s_label.item(), err_s_domain.item()))
            
            print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(
                sum(train_correct_list), len(source_loader.dataset),
                100. * sum(train_correct_list) / len(source_loader.dataset)))
            
            model.eval()
            with torch.no_grad():
                eval_correct = []
                eval_loss_list = []
                for batch_idx, (eval_data, eval_label) in enumerate(eval_loader):
                    eval_data = eval_data.to(self.device)
                    eval_label = eval_label.to(self.device)

                    # feature = feature_extractor(eval_data)
                    # class_logits = label_predictor(feature)
                    class_logits, _ = model(input_data=eval_data, alpha=alpha)
                    eval_loss_list.append(loss_class(class_logits, eval_label).item())
                    eval_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == eval_label).item())

                print('Eval set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        np.mean(eval_loss_list), sum(eval_correct), len(eval_loader.dataset),
                        100. * sum(eval_correct) / len(eval_loader.dataset)))

                test_correct = []
                test_loss_list = []
                for batch_idx, (test_data, test_label) in enumerate(test_loader):
                    test_data = test_data.to(self.device)
                    test_label = test_label.to(self.device)

                    # feature = feature_extractor(test_data)
                    # class_logits = label_predictor(feature)
                    class_logits, _ = model(input_data=test_data, alpha=alpha)
                    test_loss_list.append(loss_class(class_logits, test_label).item())
                    test_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == test_label).item())

                print('Test set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                        np.mean(test_loss_list), sum(test_correct), len(test_loader.dataset),
                        100. * sum(test_correct) / len(test_loader.dataset)))
            
            # if epoch % 5 == 0 and epoch > 0:
            save_path = os.path.join(self.ckpt_dir, f'{epoch}.pth')
            self.save_checkpoint_github(save_path, model, optimizer, epoch)

    def upper(self, source_loader, target_loader, eval_loader, test_loader):
        model = CNNModel()
        model.to(self.device)
        
        # pdb.set_trace()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_class = torch.nn.NLLLoss()
        loss_domain = torch.nn.NLLLoss()

        for epoch in range(self.max_epoch):
            model.train()
            len_dataloader = len(target_loader)
            # data_source_iter = iter(source_loader)
            data_target_iter = iter(target_loader)

            loss_domain_list = []
            loass_class_list = []
            train_correct_list = []
            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / self.max_epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                # s_img, s_label = data_source_iter.next()
                # s_img, s_label = s_img.to(self.device), s_label.to(self.device)

                # model.zero_grad()
                # batch_size = len(s_label)
                # domain_label = torch.zeros(batch_size).long().to(self.device)
                # class_output, domain_output = model(input_data=s_img, alpha=alpha)
                
                # err_s_label = loss_class(class_output, s_label)
                # err_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                data_target = data_target_iter.next()
                t_img, t_label = data_target
                t_img = t_img.to(self.device)
                t_label = t_label.to(self.device)

                batch_size = len(t_img)

                domain_label = torch.ones(batch_size).long().to(self.device)
                class_output, domain_output = model(input_data=t_img, alpha=alpha)
                err_t_label = loss_class(class_output, t_label)
                err_t_domain = loss_domain(domain_output, domain_label)
                # err = err_t_domain + err_s_domain + err_s_label
                err = err_t_domain + err_t_label
                err.backward()
                optimizer.step()

                loss_domain_list.append(err_t_domain.item())
                loass_class_list.append(err_t_label.item())
                train_correct_list.append(torch.sum(torch.argmax(class_output, dim=1) == t_label).item())

                if i % 30 == 0:
                    print('\r epoch: %d, [iter: %d / all %d], err_t_label: %f, err_t_domain: %f' \
                    % (epoch, i + 1, len_dataloader, err_t_label.item(), err_t_domain.item()))
            
            print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(
                sum(train_correct_list), len(source_loader.dataset),
                100. * sum(train_correct_list) / len(source_loader.dataset)))
            
            model.eval()
            with torch.no_grad():
                eval_correct = []
                eval_loss_list = []
                for batch_idx, (eval_data, eval_label) in enumerate(eval_loader):
                    eval_data = eval_data.to(self.device)
                    eval_label = eval_label.to(self.device)

                    # feature = feature_extractor(eval_data)
                    # class_logits = label_predictor(feature)
                    class_logits, _ = model(input_data=eval_data, alpha=alpha)
                    eval_loss_list.append(loss_class(class_logits, eval_label).item())
                    eval_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == eval_label).item())

                print('Eval set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        np.mean(eval_loss_list), sum(eval_correct), len(eval_loader.dataset),
                        100. * sum(eval_correct) / len(eval_loader.dataset)))

                test_correct = []
                test_loss_list = []
                for batch_idx, (test_data, test_label) in enumerate(test_loader):
                    test_data = test_data.to(self.device)
                    test_label = test_label.to(self.device)

                    # feature = feature_extractor(test_data)
                    # class_logits = label_predictor(feature)
                    class_logits, _ = model(input_data=test_data, alpha=alpha)
                    test_loss_list.append(loss_class(class_logits, test_label).item())
                    test_correct.append(torch.sum(torch.argmax(class_logits, dim=1) == test_label).item())

                print('Test set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                        np.mean(test_loss_list), sum(test_correct), len(test_loader.dataset),
                        100. * sum(test_correct) / len(test_loader.dataset)))
            
            if epoch % 5 == 0 and epoch > 0:
                save_path = os.path.join(self.ckpt_dir, f'{epoch}.pth')
                self.save_checkpoint_github(save_path, model, optimizer, epoch)





            

        