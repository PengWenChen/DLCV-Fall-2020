import torchvision.models as models
from torch.utils.data import DataLoader
import torch
import pdb
import numpy as np
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
import sys
import csv

from utils import *
from model import *

# # fix random seeds for reproducibility
# SEED = 65
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(SEED)
# np.random.seed(SEED)

class STMT():
    def __init__(self, config):
        self.config = config
        self.lr = 1e-4
        self.max_epoch = 20
        self.batch_size = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Fbeta(self, output, gt, beta=2):
        re = recall_score(gt, output,average='binary')
        ps = precision_score(gt,output,average='binary')
        return (1+beta**2)*(re*ps)/(re+(beta**2)*ps+sys.float_info.epsilon)
    
    def save_checkpoint(self, checkpoint_path, save_model_name, model, optimizer, epoch):
        print('\nStart saving ...')
        state = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,}
        model_name = save_model_name+f'_{epoch}.pth'
        save_path = os.path.join(checkpoint_path, model_name)
        torch.save(state, save_path, _use_new_zipfile_serialization=False) # for old version model
        print('model saved to %s\n' % save_path) 
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=False):
        state = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(state['model'])
        print('model loaded from %s\n' % checkpoint_path)

    def train_single(self, validation_split=True, validation_only=False):
        to_train, to_val = split_data(self.config.csv_path, split=validation_split)
            
        val_dataset = MedicalDataset_single(self.config.img_dir, patients_id_list=to_val, csv_path=self.config.csv_path, val=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        train_dataset = MedicalDataset_single(self.config.img_dir, patients_id_list=to_train, csv_path=self.config.csv_path)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

        if self.config.model_which=='Res18':
            resnet18 = models.resnet18(pretrained=True)
            model = ModelSingle(resnet18)
        elif self.config.model_which=='resnet34':
            resnet34 = models.resnet34(pretrained=True)
            model = ModelSingle(resnet34)
        elif self.config.model_which=='mobilenet_v2':
            mobilenet_v2 = models.mobilenet_v2(pretrained=True)
            model = ModelSingle(mobilenet_v2)
        
        if validation_only: # Load the model and calculate validation score without training
            self.load_checkpoint(self.config.model_dir, model)

        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        criterion = AsymmetricLossOptimized()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        iteration = 0
        for epoch in range(self.max_epoch):
            train_loss_list = []
            train_fbeta_list = []

            model.train()
            for batch_idx, (img, label) in enumerate(train_loader):
                # pdb.set_trace()
                if validation_only:
                    break
                img = img.to(self.device)
                label = label.to(self.device)
                pred = model(img)

                weight = torch.tensor([0.1, 0.9])
                weight_ = weight[label.data.view(-1).long()].view_as(label).to(self.device)
                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred=torch.sigmoid(pred)
                f_pred = pred.view(-1).cpu().detach().numpy().round()
                f_label = label.view(-1).cpu().numpy().round()
                fbeta = self.Fbeta(f_pred, f_label)

                if iteration % 50 == 0 :
                    print('epoch {}, train {}/{}, loss={:.4f} fbeta={:.4f}'
                                .format(epoch, batch_idx, len(train_loader), loss.item(), fbeta))

                iteration += 1
                train_loss_list.append(loss.item())
                train_fbeta_list.append(fbeta)

            if not validation_only:
                print('\nEpoch {}, train_loss={:.4f} fbeta={:.4f}'
                            .format(epoch, np.mean(train_loss_list), np.mean(train_fbeta_list)))

            model.eval()
            with torch.no_grad():
                val_loss_list = []
                val_fbeta_list = []

                val_iter = 0
                for batch_idx, (img, label) in enumerate(val_loader):
                    if not validation_split and not validation_only:
                        break
                    img = img.to(self.device)
                    label = label.to(self.device)
                    pred = model(img)

                    loss = criterion(pred, label)

                    pred = torch.sigmoid(pred)
                    f_pred = pred.view(-1).cpu().detach().numpy().round()
                    f_label = label.view(-1).cpu().numpy().round()
                    fbeta = self.Fbeta(f_pred, f_label)

                    if batch_idx==0:
                        all_f_pred = f_pred
                        all_f_label = f_label
                    else:
                        all_f_pred = np.concatenate((all_f_pred,f_pred))
                        all_f_label = np.concatenate((all_f_label,f_label))

                    if val_iter % 10 == 0 :
                        print('epoch {}, val {}/{}, val_loss={:.4f} val_fbeta={:.4f}'
                                    .format(epoch, batch_idx, len(val_loader), loss.item(), fbeta))
                    
                    val_iter += 1
                    val_loss_list.append(loss.item())
                    val_fbeta_list.append(fbeta)

                if validation_split or validation_only:
                    print('\ntrue val fbeta', self.Fbeta(all_f_pred, all_f_label))
                    print('Epoch {}, val_loss={:.4f} fbeta={:.4f}'
                                .format(epoch, np.mean(val_loss_list), np.mean(val_fbeta_list)))
            if validation_only:
                break
            else:
                scheduler.step()
                self.save_checkpoint(self.config.checkpoint_path, self.config.model_name, model, optimizer, epoch)
    
    def finetune_with_multi(self, validation_split=True, validation_only=False):
        to_train, to_val = split_data(self.config.csv_path, split=validation_split)

        train_dataset = MedicalDataset_multi(self.config.img_dir, patients_id_list=to_train, csv_path=self.config.csv_path)
        val_dataset = MedicalDataset_multi(self.config.img_dir, patients_id_list=to_val, csv_path=self.config.csv_path)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=20)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20)

        if self.config.model_which=='Res18':
            resnet18 = models.resnet18(pretrained=True)
            model = ModelMulti(resnet18)
        elif self.config.model_which=='resnet34':
            resnet34 = models.resnet34(pretrained=True)
            model = ModelMulti(resnet34)
        elif self.config.model_which=='mobilenet_v2':
            mobilenet_v2 = models.mobilenet_v2(pretrained=True)
            model = ModelMulti(mobilenet_v2)
        
        self.load_checkpoint(self.config.model_dir, model)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = AsymmetricLossOptimized()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        iteration = 0
        
        if not validation_only:
            for epoch in range(self.max_epoch):
                train_loss_list = []
                train_fbeta_list = []

                model.train()
                for batch_idx, (img, label) in enumerate(train_loader):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    weight = torch.tensor([0.1, 0.3, 0.5, 0.3, 0.1]).unsqueeze(0).unsqueeze(2).repeat(img.shape[0], 1, 5)
                    weight = weight.to(self.device)
                    pred = model(img ,weight)

                    loss = criterion(pred, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pred=torch.sigmoid(pred)
                    f_pred = pred.view(-1).cpu().detach().numpy().round()
                    f_label = label.view(-1).cpu().numpy().round()
                    
                    fbeta = self.Fbeta(f_pred, f_label)

                    train_loss_list.append(loss.item())
                    train_fbeta_list.append(fbeta)

                    if iteration % 10 == 0:
                        print('Epoch: {},\titeration: {}, train_loss: {:.4f},\t train_fbeta: {} '.format(
                                epoch, iteration, np.mean(train_loss_list), np.mean(train_fbeta_list) ))

                    if iteration % 10 == 0 and validation_split:
                        model.eval()
                        with torch.no_grad():
                            val_loss_list = []
                            val_fbeta_list = []

                            val_iter = 0
                            for batch_idx, (img, label) in enumerate(val_loader):
                                img = img.to(self.device)
                                label = label.to(self.device)
                                weight = torch.tensor([0.1, 0.3, 0.5, 0.3, 0.1]).unsqueeze(0).unsqueeze(2).repeat(img.shape[0], 1, 5)
                                weight = weight.to(self.device)
                                pred = model(img, weight)

                                loss = criterion(pred, label)
                                pred=torch.sigmoid(pred)
                                f_pred = pred.view(-1).cpu().detach().numpy().round()
                                f_label = label.view(-1).cpu().numpy().round()
                                if batch_idx==0:
                                    all_f_pred=f_pred
                                    all_f_label=f_label
                                else:
                                    all_f_pred=np.concatenate((all_f_pred,f_pred))
                                    all_f_label=np.concatenate((all_f_label,f_label))
                                fbeta = self.Fbeta(f_pred, f_label)      

                                val_iter += 1
                                val_loss_list.append(loss.item())
                            print('Epoch: {},\titeration: {}, valid_loss: {:.4f},\t true val fbeta: {} \n'.format(
                                epoch, iteration, np.mean(val_loss_list), self.Fbeta(all_f_pred, all_f_label) ))

                    if iteration % 100 == 0:
                        self.save_checkpoint(self.config.checkpoint_path, self.config.model_name + '_iter', model, optimizer, iteration)                
                    
                    iteration += 1

                scheduler.step()
        else:
            model.eval()
            with torch.no_grad():
                val_loss_list = []
                val_fbeta_list = []

                val_iter = 0
                for batch_idx, (img, label) in enumerate(val_loader):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    weight = torch.tensor([0.1, 0.3, 0.5, 0.3, 0.1]).unsqueeze(0).unsqueeze(2).repeat(img.shape[0], 1, 5)
                    weight = weight.to(self.device)
                    pred = model(img, weight)

                    loss = criterion(pred, label)
                    pred=torch.sigmoid(pred)
                    f_pred = pred.view(-1).cpu().detach().numpy().round()
                    f_label = label.view(-1).cpu().numpy().round()
                    if batch_idx==0:
                        all_f_pred=f_pred
                        all_f_label=f_label
                    else:
                        all_f_pred=np.concatenate((all_f_pred,f_pred))
                        all_f_label=np.concatenate((all_f_label,f_label))
                    fbeta = self.Fbeta(f_pred, f_label)      

                    val_iter += 1
                    val_loss_list.append(loss.item())
                print(f'{val_iter}/{len(val_loader)}', end='\r')
                print('\nvalid_loss: {:.4f},\t true val fbeta: {} '.format(
                    np.mean(val_loss_list), self.Fbeta(all_f_pred, all_f_label) ))

    def test(self, image_mode="multi"):
        if image_mode=="multi":
            test_dataset = MedicalDataset_Sequence_Test(self.config.test_dir, test=True)
        elif image_mode=="single":
            test_dataset = MedicalDataset_single(self.config.test_dir, test=True)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        if self.config.model_which=='Res18':
            pretrain = models.resnet18(pretrained=True)
        elif self.config.model_which=='resnet34':
            pretrain = models.resnet34(pretrained=True)
        elif self.config.model_which=='mobilenet_v2':
            pretrain = models.mobilenet_v2(pretrained=True)
        
        if image_mode=="multi":
            model = ModelMultiEnsemble(pretrain)
        elif image_mode=="single":
            model = ModelSingle(pretrain)

        self.load_checkpoint(self.config.model_dir, model)
        model.to(self.device)

        model.eval()
        with torch.no_grad():
            record_photo_name = []
            record_predict = []
            print("Predicting ...")
            for batch_idx, (photo_name, img) in enumerate(test_loader):
                img = img.to(self.device)

                if image_mode=="multi":
                    img = img.squeeze()
                    weight = torch.tensor([0.1, 0.3, 0.5, 0.3, 0.1]).unsqueeze(0).unsqueeze(2).repeat(img.shape[0], 1, 5)
                    weight = weight.to(self.device)
                    pred = model(img, weight)
                    record_photo_name += [i[0] for i in photo_name]
                elif image_mode=="single":
                    img = img.to(self.device)
                    pred = model(img)
                    record_photo_name += [i for i in photo_name]

                pred = torch.sigmoid(pred)
                record_predict += pred.view(-1).cpu().detach().numpy().round().astype(int).tolist()
                print(f'{batch_idx}/{len(test_loader)}', end='\r')

        with open(self.config.output_csv_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dirname', 'ID', 'ich', 'ivh', 'sah', 'sdh', 'edh'])

            for i in range(len(record_photo_name)):
                patient_id = record_photo_name[i].split('_')[0]
                patient_id = 'ID_' + patient_id
                writer.writerow([patient_id, record_photo_name[i], 
                    record_predict[i*5], record_predict[i*5+1], 
                    record_predict[i*5+2], record_predict[i*5+3], record_predict[i*5+4]])
        print("Done")
        
    def test_ensemble(self):
        # test_dataset = MedicalDataset_triple(self.config.test_dir, test=True)
        test_dataset = MedicalDataset_Sequence_Test(self.config.test_dir, test=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        if self.config.model_which=='Res18':
            resnet18 = models.resnet18(pretrained=True)
            model = ModelMultiEnsemble(resnet18)
        elif self.config.model_which=='resnet34':
            resnet34 = models.resnet34(pretrained=True)
            model1 = ModelMultiEnsemble(resnet34)
            model2 = ModelMultiEnsemble(resnet34)
            model3 = ModelMultiEnsemble(resnet34)
        elif self.config.model_which=='mobilenet_v2':
            resnet18 = models.mobilenet_v2(pretrained=True)
            model = ModelMultiEnsemble(resnet18)

        self.load_checkpoint(self.config.model1, model1) # 0.77221
        self.load_checkpoint(self.config.model2, model2)     # 0.77176
        self.load_checkpoint(self.config.model3, model3)   # 0.77745

        model1.to(self.device)
        model2.to(self.device)
        model3.to(self.device)

        model1.eval()
        model2.eval()
        model3.eval()
        with torch.no_grad():
            record_photo_name = []
            record_predict = []
            print("\nPredicting with multi-image weight: [0.1, 0.3, 0.5, 0.3, 0.1]...")
            for batch_idx, (photo_name, img) in enumerate(test_loader):
                img = img.squeeze()
                weight = torch.tensor([0.1, 0.3, 0.5, 0.3, 0.1]).unsqueeze(0).unsqueeze(2).repeat(img.shape[0], 1, 5)
                weight = weight.to(self.device)
                img = img.to(self.device)
                pred1 = model1(img, weight)
                pred2 = model2(img, weight)
                pred3 = model3(img, weight)
                pred = (pred1 + pred2 + pred3) / 3 
                pred = torch.sigmoid(pred)
                record_photo_name += [i[0] for i in photo_name]
                record_predict += pred.view(-1).cpu().detach().numpy().round().astype(int).tolist()
                print(f'{batch_idx}/{len(test_loader)}', end='\r')

        with open(self.config.output_csv_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dirname', 'ID', 'ich', 'ivh', 'sah', 'sdh', 'edh'])

            for i in range(len(record_photo_name)):
                patient_id = record_photo_name[i].split('_')[0]
                patient_id = 'ID_' + patient_id
                writer.writerow([patient_id, record_photo_name[i], 
                    record_predict[i*5], record_predict[i*5+1], 
                    record_predict[i*5+2], record_predict[i*5+3], record_predict[i*5+4]])
        print("Done")