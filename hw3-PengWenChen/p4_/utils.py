from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
import csv
import os
import pdb

class DigitDataset(Dataset):
    def __init__(self, root, csv_path):
        self.root = root
        self.img_file_list = []
        self.label_list = []

        with open(csv_path, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                try:
                    self.label_list.append(int(row['label']))
                except:
                    pdb.set_trace()
        try:
            for file_name in sorted(os.listdir(self.root)):
                self.img_file_list.append(file_name)
        except:
            pdb.set_trace()
        
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.44, 0.44, 0.44], std=[0.19, 0.19, 0.19])
        ])

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.root, self.img_file_list[item])).convert('RGB')
        # img = Image.open(os.path.join(self.root, self.img_file_list[item]))
        img = self.transform(img)
        label = self.label_list[item]
        return img, label

    def __len__(self):
        return len(self.img_file_list)

class DigitDataset_test_usps_mnistm(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_file_list = []

        try:
            for file_name in sorted(os.listdir(self.root)):
                self.img_file_list.append(file_name)
        except:
            pdb.set_trace()

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.44, 0.44, 0.44], std=[0.19, 0.19, 0.19])
        ])

    def __getitem__(self, item):
        file_name = self.img_file_list[item]
        img = Image.open(os.path.join(self.root, self.img_file_list[item])).convert('RGB')
        img = self.transform(img)
        return file_name, img

    def __len__(self):
        return len(self.img_file_list)

class DigitDataset_test_svhn(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_file_list = []

        try:
            for file_name in sorted(os.listdir(self.root)):
                self.img_file_list.append(file_name)
        except:
            pdb.set_trace()

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

    def __getitem__(self, item):
        file_name = self.img_file_list[item]
        img = Image.open(os.path.join(self.root, self.img_file_list[item]))
        img = self.transform(img)
        return file_name, img

    def __len__(self):
        return len(self.img_file_list)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		size = m.weight.size()
		m.weight.data.normal_(0.0, 0.1)
		m.bias.data.fill_(0)

def lr_scheduler(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer

def exp_lr_scheduler(optimizer, epoch, init_lr, lrd, nevals):
    """Implements torch learning reate decay with SGD"""
    lr = init_lr / (1 + nevals*lrd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer