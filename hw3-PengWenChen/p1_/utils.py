from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch.nn as nn
import torch
from PIL import Image
import os
import pdb

class FaceDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.file_list = []
        
        for file_name in sorted(os.listdir(root)):
            self.file_list.append(file_name)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
    def __getitem__(self, item):
        img = os.path.join(self.root, self.file_list[item])
        img = Image.open(img)
        img = self.transform(img)

        return img
    def __len__(self):
        return len(self.file_list)

def VAE_loss(criterion, recon_x, x, mu, log_var, lamb=1):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    # loss = F.mse_loss(recon_x, x, reduction='mean') / output.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return mse + lamb*KLD, mse, KLD

# def VAE_loss(recon_x, x, mu, logvar):
# #     BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    
#     loss = nn.L1Loss(size_average=False)
# #     MSE = F.mse_loss(recon_x, x, size_average=False)
#     pdb.set_trace()
#     l1_loss = loss(recon_x, x)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return l1_loss + KLD, l1_loss, KLD
