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

from p3_.utils import DigitDataset, DigitDataset_test
from p3_.p3_run import DANN

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='train')
parser.add_argument("--restore", default=False)
parser.add_argument("--checkpoint_dir", default="./model_test")

# Train
parser.add_argument("--max_epoch", default=50)
parser.add_argument("--lr", default=1e-4) # svhn=1e-4
parser.add_argument("--batch_size", default=256) # svhn=256
parser.add_argument("--lamb", default=0.1) # svhn=0.5

# Test
parser.add_argument("--target_test_dir") # $1
parser.add_argument("--target_name", default="mnistm") # $2
parser.add_argument("--csv_output_dir") # $3
parser.add_argument("--tsne", default=False)
args = parser.parse_args()

root = "./hw3_data/digits/"
target_dir = os.path.join(root, args.target_name, "train")
target_label = os.path.join(root, args.target_name, "train.csv")

if args.target_test_dir:
    test_dir = args.target_test_dir
else:
    test_dir = os.path.join(root, args.target_name, "test")
test_label = os.path.join(root, args.target_name, "test.csv")

if args.target_name=="mnistm":
    source_dir = "./hw3_data/digits/usps/train"
    source_label = "./hw3_data/digits/usps/train.csv"
    eval_dir = "./hw3_data/digits/usps/test"
    eval_label = "./hw3_data/digits/usps/test.csv"

elif args.target_name=="svhn":
    source_dir = "./hw3_data/digits/mnistm/train"
    source_label = "./hw3_data/digits/mnistm/train.csv"
    eval_dir = "./hw3_data/digits/mnistm/test"
    eval_label = "./hw3_data/digits/mnistm/test.csv"

elif args.target_name=="usps":
    source_dir = "./hw3_data/digits/svhn/train"
    source_label = "./hw3_data/digits/svhn/train.csv"
    eval_dir = "./hw3_data/digits/svhn/test"
    eval_label = "./hw3_data/digits/svhn/test.csv" 

if args.mode=="train":
    print(f"Start training on {args.target_name}")

    dann = DANN(args)
    source_dataset = DigitDataset(source_dir, source_label)
    target_dataset = DigitDataset(target_dir, target_label)
    eval_dataset = DigitDataset(eval_dir, eval_label)
    test_dataset = DigitDataset(test_dir, test_label)

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    dann.train_2_github(source_loader, target_loader, eval_loader, test_loader)
    # dann.lower(source_loader, target_loader, eval_loader, test_loader)
    # dann.upper(source_loader, target_loader, eval_loader, test_loader)

elif args.tsne:
    dann = DANN(args)
    test_dataset = DigitDataset(test_dir, test_label)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    eval_dataset = DigitDataset(eval_dir, eval_label)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.target_name=="mnistm":
        model_dir = "./p3_mnistm_best.pth?dl=1"
        img1_dir = "./mnistm_1.jpg"
        img2_dir = "./mnistm_2.jpg"
    elif args.target_name=="svhn":
        model_dir = "./p3_svhn_best.pth?dl=1"
        img1_dir = "./svhn_1.jpg"
        img2_dir = "./svhn_2.jpg"
    elif args.target_name=="usps":
        model_dir = "./p3_usps_best.pth?dl=1"
        img1_dir = "./usps_1.jpg"
        img2_dir = "./usps_2.jpg"

    dann.tsne(args.target_name, test_loader, eval_loader, model_dir, img1_dir, img2_dir)

elif args.mode=="test":
    dann = DANN(args)
    test_dataset = DigitDataset_test(args.target_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.target_name=="mnistm":
        model_dir = "./p3_mnistm_best.pth?dl=1" 
    elif args.target_name=="svhn":
        model_dir = "./p3_svhn_best.pth?dl=1"
    elif args.target_name=="usps":
        model_dir = "./p3_usps_best.pth?dl=1"


    dann.test(test_loader, model_dir, args.csv_output_dir)