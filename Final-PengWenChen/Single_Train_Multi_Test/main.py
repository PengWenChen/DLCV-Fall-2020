from stmt_run import STMT
import torch
import random
import numpy as np
import argparse
import time
import pdb

parser = argparse.ArgumentParser()

parser.add_argument("--mode",
    help="train_single, finetune_with_multi, test_multi_ensemble, test_multi, test_single")

parser.add_argument("--seed", default=65, help="Please choose a seed, you can either use 123, 65, 413")
# For training
parser.add_argument("--img_dir", default="../Blood_data/train")
parser.add_argument("--csv_path", default="../Blood_data/train.csv")
parser.add_argument("--checkpoint_path", default="../model", help="Please give a model saving path.")
parser.add_argument("--model_name", default="test", help="Please give a model name for saving, i.e.'Resnet18_single'") 

parser.add_argument("--model_dir", default="../model/ann_res_seed65_9.pth")
parser.add_argument("--validation_split", default=False, help="True for training whole data, False for split validation set")
parser.add_argument("--validation_only", default=False, help="True for only see validation score of whole training data, without training")
parser.add_argument("--model_which", default="resnet34")

# For ensemble with multi inference
parser.add_argument("--model1", default="../model1.pth?dl=1")
parser.add_argument("--model2", default="../model2.pth?dl=1")
parser.add_argument("--model3", default="../model3.pth?dl=1")
parser.add_argument("--output_csv_name", default="test.csv")

parser.add_argument("--test_dir", default="../Blood_data/test")
args = parser.parse_args()


# fix random seeds for reproducibility
SEED = int(args.seed)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

stmt_obj = STMT(args)
if args.mode=='train_single':
    if args.validation_only:
        print("Single Validating ...")
    else:
        print("Single Training ...")
    stmt_obj.train_single(validation_split=args.validation_split, validation_only=args.validation_only)

elif args.mode=='finetune_with_multi':
    if args.validation_only:
        print("Validating on multi images after finetuning...")
    else:
        print("Finetuning with multi images ...")
    stmt_obj.batch_size = 12 # the dataset of multi images return 5 images
    stmt_obj.lr = 1e-6
    stmt_obj.finetune_with_multi(validation_split=args.validation_split, validation_only=args.validation_only)

elif args.mode=='test_multi_ensemble':
    print(f'Start multi-image-ensemble, testing on {args.test_dir}')
    st = time.time()
    stmt_obj.test_ensemble()
    ed = time.time()
    print(f"Total run time = {ed-st} seconds.")

elif args.mode=='test_multi':
    print(f'Start multi-image test, testing on {args.test_dir}')
    st = time.time()
    stmt_obj.test(image_mode="multi")
    ed = time.time()
    print(f"Total run time = {ed-st} seconds.")

elif args.mode=='test_single':
    print(f'Start single-image test, testing on {args.test_dir}')
    st = time.time()
    stmt_obj.test(image_mode="single")
    ed = time.time()
    print(f"Total run time = {ed-st} seconds.")

