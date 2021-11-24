from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
from PIL import Image
import os
import pdb

class ClassifyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_path = []
        for img_file in sorted(os.listdir(self.root)):
            self.data_path.append(img_file)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # pdb.set_trace()
    def __getitem__(self, item):
        path = os.path.join(self.root, self.data_path[item])
        img = Image.open(path)
        img = self.transform(img)
        target = int(self.data_path[item].split('_')[0])
        return self.data_path[item], img, target
    
    def __len__(self):
        return len(self.data_path)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ClassifyDataset_for_test(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_path = []
        for img_file in sorted(os.listdir(self.root)):
            self.data_path.append(img_file)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    def __getitem__(self, item):
        path = os.path.join(self.root, self.data_path[item])
        img = Image.open(path)
        img = self.transform(img)
        return self.data_path[item], img
    def __len__(self):
        return len(self.data_path)

class Vggfcn(nn.Module):
    def __init__(self, pretrain=False):
        super(Vggfcn, self).__init__()
        self.pretrain = pretrain
        self.layer1 = pretrain.features
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(6144, 2048),
            nn.Linear(2048, 256),
            nn.Linear(256, 50),
        )

    def forward(self, input):
        x = input
        x = self.layer1(x) #[32, 512, 1, 1]
        x = self.conv1(x) 
        x = self.up(x) # 512, 2, 2
        x = self.conv2(x)
        x = self.up(x) # 256, 4, 4
        x2 = self.conv3(x)
        x3 = torch.cat([x, x2], dim=1)
        x3 = x3.view(x3.shape[0], -1)
        output = self.fc(x3)
        return output

