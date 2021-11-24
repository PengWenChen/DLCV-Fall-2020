from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import OrderedDict
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pdb
import numpy as np
import random

def split_data(csv_path, split=True):
    df = pd.read_csv(csv_path)
    patients_id = df.dirname
    patients_id_list = list(dict.fromkeys(patients_id))
    random.shuffle(patients_id_list)
    if split:
        to_train = patients_id_list[:1400]
        to_val = patients_id_list[1400:]
        print("Data Spliting...")
        return to_train, to_val
    else:
        return patients_id_list, patients_id_list[1400:]

# Train on sequence data Ann
class MedicalDataset_Sequence_Train(Dataset):
    def __init__(self, img_dir, patients_id_list, csv_path, val=False):
        self.patients_id_list = patients_id_list
        self.img_dir = img_dir
        self.csv = pd.read_csv(csv_path)

        if not val:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(0, translate=(0.2,0.2)),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
 


    def __getitem__(self, idx):
        patient_id = self.patients_id_list[idx]
        patient_num = patient_id.split('_')[1]
        photo_dir = os.path.join(self.img_dir, patient_id)
        # print(photo_dir)
        img_list = []
        label_list = []
        for photo in range(self.csv.loc[self.csv['dirname']==patient_id].shape[0]):
            file_name = (patient_num+'_'+str(photo)+'.jpg')
            img = Image.open(os.path.join(photo_dir, file_name)).convert('RGB')
            img = self.transform(img)
            img_list.append(img)
            label_list.append(self.get_label(file_name))
        img_list = torch.stack(img_list, dim=0)
        # pdb.set_trace()

        label_list = np.array(label_list)
        label_list = torch.tensor(label_list, dtype=torch.float32)



        assert len(img_list) == len(label_list)
        return img_list, label_list #[ [0,1,0,1,0], [0,0,0,0,0], [], []]
    
    def __len__(self):
        return len(self.patients_id_list)
    
    def get_label(self, photo):
        row = self.csv.loc[self.csv['ID'] == photo]
        label = [row.ich.values[0], row.ivh.values[0], row.sah.values[0], row.sdh.values[0], row.edh.values[0]]
        # label = torch.tensor(label, dtype=torch.int)
        return label

# Test on sequence data Ann
class MedicalDataset_Sequence_Test(Dataset):
    def __init__(self, img_dir,  test):
        # self.patients_id_list = patients_id_list
        self.img_dir = img_dir
        self.patients_id_list = []

        for patients_id in sorted(os.listdir(self.img_dir)):
            self.patients_id_list.append(patients_id)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        # pdb.set_trace()

    def __getitem__(self, idx):
        patient_id = self.patients_id_list[idx]
        patient_num = patient_id.split('_')[1]
        photo_dir = os.path.join(self.img_dir, patient_id)
        # print(photo_dir)
        img_list = []
        photo_name_list = []
        num = len(os.listdir(photo_dir))
        for i in range(num):
            photo = patient_num + '_' + str(i) + '.jpg'
            img = Image.open(os.path.join(photo_dir, photo)).convert('RGB')
            img = self.transform(img)
            img_list.append(img)
            photo_name_list.append(photo)
        img_list = torch.stack(img_list, dim=0)
        
        return photo_name_list, img_list #[ [0,1,0,1,0], [0,0,0,0,0], [], []]
    
    def __len__(self):
        return len(self.patients_id_list)

# Train/Validation/Test 
class MedicalDataset_single(Dataset):
    def __init__(self, img_dir, patients_id_list = 0, csv_path = 0, test=False, val=False):
        self.val = val
        self.test = test
        self.patients_id_list = patients_id_list
        self.img_dir = img_dir

        self.photo_path = []
        if test:
            for root, dirs, files in os.walk(img_dir):
                for f in files:
                    fullpath = os.path.join(root, f)
                    self.photo_path.append(fullpath)
        else:
            self.csv = pd.read_csv(csv_path)
            for patients_id in patients_id_list:
                for photo in sorted(os.listdir(os.path.join(img_dir, patients_id))):
                    self.photo_path.append(os.path.join(img_dir, patients_id, photo))

        self.transform_train = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(0, translate=(0.2,0.2)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            ])

    def __getitem__(self, idx):
        img = Image.open(self.photo_path[idx]).convert('RGB')
        if self.test==False and self.val==False:
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        if self.test:
            photo_name = self.photo_path[idx].split('/')[-1]
            return photo_name, img
        else:
            label = self.get_label(idx)
            return img, label
    
    def __len__(self):
        return len(self.photo_path)
    
    def get_label(self, idx):
        photo = self.photo_path[idx].split('/')[-1]
        row = self.csv.loc[self.csv['ID'] == photo]
        label = [row.ich.values[0], row.ivh.values[0], row.sah.values[0], row.sdh.values[0], row.edh.values[0]]
        label = torch.tensor(label, dtype=torch.float32)
        return label

# Train/Validation/Test Lewis
class MedicalDataset_multi(Dataset):
    def __init__(self, img_dir, patients_id_list = 0, csv_path = 0, test=False, val=False):
        self.val = val
        self.test = test
        self.patients_id_list = patients_id_list
        self.img_dir = img_dir

        self.photo_path = []
        if test:
            for root, dirs, files in os.walk(img_dir):
                for f in files:
                    fullpath = os.path.join(root, f)
                    self.photo_path.append(fullpath)
        else:
            self.csv = pd.read_csv(csv_path)
            for patients_id in patients_id_list:
                for photo in sorted(os.listdir(os.path.join(img_dir, patients_id))):
                    self.photo_path.append(os.path.join(img_dir, patients_id, photo))

        self.transform_train = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(0, translate=(0.2,0.2)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        photo_dir=self.photo_path[idx]
        photo_dir=photo_dir.split('/')[0]+'/'+photo_dir.split('/')[1]+'/'+photo_dir.split('/')[2]+'/'+photo_dir.split('/')[3]+'/'

        patient_num=photo_dir.split('/')[-2].split('_')[1]

        seq=self.photo_path[idx]
        image_idx=int(seq.split('_')[-1].split('.')[0])
        
        id0 = max(0, image_idx-2)
        id1 = max(0, image_idx-1)
        id2 = image_idx
        id3 = min(len(os.listdir(photo_dir))-1, image_idx+1)
        id4 = min(len(os.listdir(photo_dir))-1, image_idx+2)

        img0 = Image.open(os.path.join(photo_dir, patient_num+'_'+str(id0)+'.jpg')).convert('RGB')
        img1 = Image.open(os.path.join(photo_dir, patient_num+'_'+str(id1)+'.jpg')).convert('RGB')
        img2 = Image.open(os.path.join(photo_dir, patient_num+'_'+str(id2)+'.jpg')).convert('RGB')
        img3 = Image.open(os.path.join(photo_dir, patient_num+'_'+str(id3)+'.jpg')).convert('RGB')
        img4 = Image.open(os.path.join(photo_dir, patient_num+'_'+str(id4)+'.jpg')).convert('RGB')

        if self.test==False and self.val==False:
            img0 = self.transform_train(img0)
            img1 = self.transform_train(img1)
            img2 = self.transform_train(img2)
            img3 = self.transform_train(img3)
            img4 = self.transform_train(img4)
        else:
            img0 = self.transform_test(img0)
            img1 = self.transform_test(img1)
            img2 = self.transform_test(img2)
            img3 = self.transform_test(img3)
            img4 = self.transform_test(img4)

        img=torch.cat((img0,img1,img2,img3,img4),dim=0)

        if self.test:
            photo_name = self.photo_path[idx].split('/')[-1]
            return photo_name, img
        else:
            label = self.get_label(idx)
            return img, label
    
    def __len__(self):
        return len(self.photo_path)
    
    def get_label(self, idx):
        photo = self.photo_path[idx].split('/')[-1]
        row = self.csv.loc[self.csv['ID'] == photo]
        label = [row.ich.values[0], row.ivh.values[0], row.sah.values[0], row.sdh.values[0], row.edh.values[0]]
        label = torch.tensor(label, dtype=torch.float32)
        return label

# asl
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
    
        return -self.loss.sum()

if __name__ == '__main__':
    random.seed(1)
    id = os.listdir('../Blood_data/train/')
    random.shuffle(id)
    
    train_data = MedicalDataset(img_dir='../Blood_data/train', patients_id_list=id, csv_path='../Blood_data/train.csv')
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
    
    label_count = np.array([0,0,0,0,0])
    
    total_image = 50492
    pos_rate = np.array([0.11706805, 0.08183475, 0.1157213 , 0.16452111, 0.03994692])
    pos_count = (pos_rate*total_image)
    max_photo = 40

    for i, (img, label) in enumerate(train_loader):
        # pdb.set_trace()
        num = label.shape[1]
            
        # label_count = label_count + label.numpy().squeeze()
    
    # pos_rate = label_count / 50492

    pdb.set_trace()
