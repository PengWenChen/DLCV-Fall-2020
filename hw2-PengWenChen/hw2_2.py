import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import argparse
import pdb
import sys
from torch.optim import lr_scheduler
import logging

logging.basicConfig(level=logging.INFO, filename='mo.log')

from hw2_2_utils import SegDataset, SegDataset_for_test, Vggfcn32, Vggfcn8, label2RGB, save
from mean_iou_evaluate import mean_iou_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(lr, epoch, train_loader, valid_loader, restore=True, save_interval=10, log_interval=30):
    pretrain = models.vgg16_bn(pretrained=True)
    # model = Vggfcn32(pretrain)
    model = Vggfcn8(pretrain)

    model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters, lr=1e-3, weight_decay=1e-6)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # decay LR by a factor of 0.5 every 30 epochs

    train_loss = []
    val_loss = []

    if restore:
        load_checkpoint('./model_2_2_best_v2/mnist-10_tmp.pth', model, optimizer)

    #weights = [5.49641505, 1.0, 6.95114893, 5.05255139, 19.02411563, 7.21815416, 924.02143829] #as class distribution
    #class_weights = torch.FloatTensor(weights).to(device)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    
    for ep in range(epoch):
        
        iteration = 0
        loss_epoch = []
        mo_epoch = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device, dtype=torch.long)
            output = model(data)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.item())

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            iteration += 1

            output = torch.argmax(output, dim=1)
            
            if batch_idx==0:
                masks = output.cpu().numpy()
                label = target.cpu().numpy()
            else:
                masks = np.concatenate((masks, output.cpu().numpy()), axis=0)
                label = np.concatenate((label, target.cpu().numpy()), axis=0)

        mo_epoch.append(mean_iou_score(masks, label))
        scheduler.step()
        print(f"training avg epoch loss: {np.mean(loss_epoch)}")
        print(f"training avg epoch mo:   {np.mean(mo_epoch)}")
        logging.info(f"training avg epoch loss: {np.mean(loss_epoch)}")
        logging.info(f"training avg epoch mo:   {np.mean(mo_epoch)}")

        if ep > 10:
            save_interval=1
        if ep % save_interval == 0 and ep > 0:
            save_checkpoint('./model_2_2_best/mnist-%i.pth' % ep, model, optimizer)
        
        train_loss.append(np.mean(loss_epoch))
        val_epoch_loss = test(model, valid_loader)
        val_loss.append(val_epoch_loss)

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train loss', 'Validation loss'])
        plt.savefig("sphere_loss.png")
        plt.clf()
    # save the final model
    save_checkpoint('./model_2_2_best/mnist-final.pth' , model, optimizer)

def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    
    mean_iou = []
    val_epoch_loss = []

    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device, dtype=torch.long)
            output = model(data)
            loss = criterion(output, target).item()
            test_loss += loss # sum up batch loss

            output = torch.argmax(output, dim=1)
            
            if batch_idx==0:
                masks = output.cpu().numpy()
                label = target.cpu().numpy()
            else:
                masks = np.concatenate((masks, output.cpu().numpy()), axis=0)
                label = np.concatenate((label, target.cpu().numpy()), axis=0)

        mean_iou.append(mean_iou_score(masks, label))
        val_epoch_loss.append(loss)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, MO: {}\n'.format(
        test_loss, np.mean(mean_iou)))
    logging.info('\nTest set: Average loss: {:.4f}, MO: {}\n'.format(
        test_loss, np.mean(mean_iou)))
    return np.mean(val_epoch_loss)

def save_checkpoint(checkpoint_path, model, optimizer):
    print('\nStart saving ...')
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s\n' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer=False):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s\n' % checkpoint_path)

def scheduler(optimizer, lr=1e-3):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def test_2(ckpt_path, output_path, best, test_loader):
    if best:
        pretrain = models.vgg16_bn(pretrained=True)
        model = Vggfcn8(pretrain)
    else:
        pretrain = models.vgg16(pretrained=True)
        model = Vggfcn32(pretrain)

    load_checkpoint(ckpt_path, model)
    model.to(device)
    model.eval()
    # pdb.set_trace()
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            data = data.to(device)
            pred = model(data)
            pred = torch.argmax(pred, dim=1)

            pred = torch.squeeze(pred)
            output = label2RGB(pred.cpu().numpy())
            save(output, output_path, index=batch_idx)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_dir", help="train img path", default="./hw2_data/p2_data/train")
    parser.add_argument("--valid_img_dir", help="validate img path", default="./hw2_data/p2_data/validation")
    parser.add_argument("--max_epoch", default=70)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--restore", default=False)

    parser.add_argument("--data_preprocess", default=True)
    
    parser.add_argument("--best", type=bool)
    parser.add_argument("--test_img_dir", help="test img path")
    parser.add_argument("--checkpoint_path", help="model.pth path", default="./model_2_2_best/mnist-45.pth")
    parser.add_argument("--output_dir", help="path of output folder", default="./hw2_data/p2_output")
    args = parser.parse_args()

    if args.test_img_dir:
        print(f'Start testing on path: {args.test_img_dir}')
        test_set = SegDataset_for_test(args.test_img_dir)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        test_2(args.checkpoint_path, args.output_dir, args.best, test_loader)
    else:
        print('Start training')
        if args.best:
            augment=True
        train_set = SegDataset(args.train_img_dir, train=augment)
        valid_set = SegDataset(args.valid_img_dir)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
        train(args.lr, args.max_epoch, train_loader, valid_loader, args.restore)



