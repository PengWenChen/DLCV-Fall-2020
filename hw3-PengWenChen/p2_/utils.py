from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
import os

class FaceDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.file_list = []
        
        for file_name in sorted(os.listdir(root)):
            self.file_list.append(file_name)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    def __getitem__(self, item):
        img = os.path.join(self.root, self.file_list[item])
        img = Image.open(img)
        img = self.transform(img)

        return img
    def __len__(self):
        return len(self.file_list)