import argparse
from torch.utils.data import DataLoader
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
import random

from p2_.utils import FaceDataset
from p2_.p2_run import GAN

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()
parser.add_argument("--train_img_dir", default="./hw3_data/face/train")
parser.add_argument("--save_dir", default="./hw3_data/gan_generate")
parser.add_argument("--restore", default=False)
parser.add_argument("--max_epoch", default=100)
parser.add_argument("--lr", default=0.0002)
parser.add_argument("--batch_size", default=64)

parser.add_argument("--latent_dim", default=100)
parser.add_argument("--nc", default=3, 
        help="Number of channels in the training images. For color images this is 3")
parser.add_argument("--ngf", default=64,
        help="Size of feature maps in generator")
parser.add_argument("--ndf", default=64,
        help="Size of feature maps in discriminator")
parser.add_argument("--beta1", default=0.5,
        help="Beta1 hyperparam for Adam optimizers")

parser.add_argument("--test_img_dir", help="img to reconstruct", default="./hw3_data/face/test")
parser.add_argument("--test_generate_path",)
parser.add_argument("--checkpoint_dir", default="./model_gan_v1")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter('./runs/' + 'gan_v1')


if args.test_generate_path:
    print("Start generating ...")
    gan = GAN(args)
    gan.test(args.checkpoint_dir, args.test_generate_path) 
else:
    print(f"Start training on {args.train_img_dir}")
    gan = GAN(args)
    
    train_dataset = FaceDataset(args.train_img_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    gan.train(train_loader)