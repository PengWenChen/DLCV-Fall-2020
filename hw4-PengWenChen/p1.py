import argparse

from p1_.p1_run import Prototypical
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str) # train, test
parser.add_argument("--train_img", default="./hw4_data/train.csv")
parser.add_argument("--train_img_dir", default="./hw4_data/train")
parser.add_argument("--train_case", default="./hw4_data/train_case_p1_1.csv")
parser.add_argument("--train_case_gt", default="./hw4_data/train_case_gt_p1_1.csv")
parser.add_argument("--model_save_dir", default="/home/pengwenchen/Desktop/DLCV/hw4-PengWenChen/model")
parser.add_argument("--metric", default="euclidean") # "euclidean, cosine, parametric"

parser.add_argument("--test_img", default="./hw4_data/val.csv") # $1
parser.add_argument("--test_img_dir", default="./hw4_data/val") # $2
parser.add_argument("--test_case", default="./hw4_data/val_testcase.csv") # $3
parser.add_argument("--test_case_output", default="./p1_output.csv") # $4
parser.add_argument("--test_case_gt", default="./hw4_data/val_testcase_gt.csv")
parser.add_argument("--model", default="best_p1_1_final.pth?dl=1")
args = parser.parse_args()

prototypical = Prototypical(args)

if args.mode=='preprocess':
    prototypical.preprocess()
elif args.mode=='train':
    print("Start training ...")
    prototypical.train()
elif args.mode=='test':
    print("Start testing ...")
    prototypical.test()