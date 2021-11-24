import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import csv
import argparse
import pdb
from PIL import Image
import os

from hw2_1_utils import ClassifyDataset, ClassifyDataset_for_test, Identity, Vggfcn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(ckpt_path, lr, epoch, train_loader, valid_loader, restore=True, save_interval=10, log_interval=30):
    pretrain = models.vgg16(pretrained=True)
    model = Vggfcn(pretrain)

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if restore:
        load_checkpoint(ckpt_path, model, optimizer)

    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    
    for ep in range(epoch):
        iteration = 0
        for batch_idx, (file_name, data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                # pdb.set_trace()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            iteration += 1
        
        if ep % save_interval == 0 and ep > 0:
            save_checkpoint('./model_2_1/2-1-%i.pth' % ep, model, optimizer)
        test(model, valid_loader)
    
    # save the final model
    save_checkpoint('./model_2_1/2-1-final.pth' , model, optimizer)

def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for file_name, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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

def draw_tsne(ckpt_path, test_loader):
    pretrain = models.vgg16(pretrained=True)
    model = Vggfcn(pretrain)
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])
    model.fc[-1] = Identity()
    model.to(device)
    model.eval()

    X = []
    y = []

    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for _, data, target in test_loader:
            data = data.to(device)
            output = model(data)
            X.append(output.cpu().numpy()[0].tolist())
            y.append(target.numpy()[0])

    print('start tsne')
    X_tsne = TSNE(n_components=2).fit_transform(X)
    print('end tsne')

    plt.figure(figsize=(16,10))
    # pdb.set_trace()
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y)
    plt.savefig('tsne.jpg')
    # pdb.set_trace()

def test2(input_path, ckpt_path, output_path, test_loader):
    pretrain = models.vgg16(pretrained=True)
    model = Vggfcn(pretrain)

    model.to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()

    correct=0
    pred_label = []
    file_name_list = []

    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for file_name, data in test_loader:
            data = data.to(device)
            output = model(data)

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            pred_label.append(pred[0][0].item())
            file_name_list.append(file_name[0])
    
    csv_path = os.path.join(output_path, 'test_pred.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'label'])
        index=0
        for file_name in file_name_list:
            # print(file_nam e, end='\r')
            try:
                writer.writerow([file_name, pred_label[index]])
            except:
                pdb.set_trace()
            index += 1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_img_dir", help="train img path", default="./hw2_data/p1_data/train_50")
    parser.add_argument("--valid_img_dir", help="validate img path", default="./hw2_data/p1_data/val_50")
    parser.add_argument("--max_epoch", default=100)
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--restore", default=False)

    parser.add_argument("--tsne", default=True)

    parser.add_argument("--checkpoint_dir", default="./model_2_1/mnist-final.pth")
    parser.add_argument("--test_img_dir", help="test img path")
    parser.add_argument("--output_dir", help="path of output folder")
    args = parser.parse_args()

    if args.test_img_dir:
        print(f'Start testing on path: {args.test_img_dir}')
        test_set = ClassifyDataset_for_test(args.test_img_dir)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        test2(args.test_img_dir, args.checkpoint_dir, args.output_dir, test_loader)
        print(f'Saving result in {args.output_dir}')
    elif args.tsne:
        valid_set = ClassifyDataset(args.valid_img_dir)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)
        draw_tsne(args.checkpoint_dir, valid_loader)
    else:
        print('Start training')
        train_set = ClassifyDataset(args.train_img_dir)
        valid_set = ClassifyDataset(args.valid_img_dir)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
        train(args.checkpoint_dir, args.lr, args.max_epoch, train_loader, valid_loader, args.restore)



