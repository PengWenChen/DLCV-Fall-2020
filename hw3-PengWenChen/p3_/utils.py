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
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.root, self.img_file_list[item]))
        img = self.transform(img)
        label = self.label_list[item]
        return img, label

    def __len__(self):
        return len(self.img_file_list)

class DigitDataset_test(Dataset):
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, item):
        file_name = self.img_file_list[item]
        img = Image.open(os.path.join(self.root, self.img_file_list[item]))
        img = self.transform(img)
        return file_name, img

    def __len__(self):
        return len(self.img_file_list)
