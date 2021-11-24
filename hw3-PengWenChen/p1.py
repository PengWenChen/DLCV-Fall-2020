import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import pdb
# from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torchvision
from PIL import Image
from torch.autograd import Variable
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

from p1_.model import VAE
from p1_.utils import FaceDataset, VAE_loss

parser = argparse.ArgumentParser()
parser.add_argument("--train_img_dir", default="./hw3_data/face/train")
parser.add_argument("--valid_img_dir", default="./hw3_data/face/test")
parser.add_argument("--max_epoch", default=500)
parser.add_argument("--lr", default=1e-5) #1e-3
parser.add_argument("--batch_size", default=128)
parser.add_argument("--latent_dim", default=512)
parser.add_argument("--restore", default=False)

parser.add_argument("--tsne", default=False)

parser.add_argument("--test_img_dir", help="img to reconstruct", default="./hw3_data/face/test")
parser.add_argument("--test_generate_path",)
parser.add_argument("--checkpoint_dir", default="./model_vae_v2/problem1_model_v2_e90.pth") #"./model_vae_v2/90.pth"
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter('./runs/' + 'vae_ex7_100')

torch.manual_seed(38)

def train(lr, batch_size, latent_dim, train_loader, valid_loader, save_interval=10, restore=False):
    model = VAE(latent_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    mseloss = nn.MSELoss()
    # mseloss = nn.L1Loss(size_average=False)
    model.train()

    lamb = 1e-3
    iteration = 0
    for epoch in range(args.max_epoch):
        loss_list = []
        kld_list = []
        mse_list = []

        if epoch<= 30:
            lamb=1e-6
        elif epoch > 30:
            lamb += 1e-6
            if lamb > 1e-5:
                lamb = 1e-5
        # lamb = 1e-5

        for batch_idx, img in enumerate(train_loader):
            print(batch_idx, end='\r')
            img = img.to(device)
            mu, log_var, recon = model(img)

            optimizer.zero_grad()
            loss, mse, kld = VAE_loss(mseloss, recon, img, mu, log_var, lamb)
            # loss, mse, kld = VAE_loss(recon, img, mu, log_var)
            loss.backward()
            optimizer.step()

            # pdb.set_trace()
            loss_list.append(loss.item())
            kld_list.append(kld.item())
            mse_list.append(mse.item())

        # writer.add_scalar('Total Loss', np.mean(loss_list), epoch)
        # writer.add_scalar('KlD', np.mean(kld_list), epoch)
        # writer.add_scalar('MSE', np.mean(mse_list), epoch)

        print('Train Epoch: {} \tLoss: {:.8f} \tKLD: {:.8f} \tMSE: {:.8f}'.format(
                        epoch, np.mean(loss_list), np.mean(kld_list), np.mean(mse_list)))
        print('lambda: {:.8f}'.format(lamb))
        valid_loss = validate(model, valid_loader, lamb)
        scheduler.step()
        if epoch % 10 == 0 and epoch > 0:
            save_checkpoint('./model_vae_v7/%i.pth' % epoch, model, optimizer)

def validate(model, test_loader, lamb):
    mseloss = nn.MSELoss()
    # mseloss = nn.L1Loss(size_average=False)
    model.eval()

    eval_loss = []
    with torch.no_grad():
        for batch_idx, img in enumerate(test_loader):
            img = img.to(device)
            mu, log_var, recon = model(img)

            loss, _, _ = VAE_loss(mseloss, recon, img, mu, log_var, lamb)
            # loss, _, _ = VAE_loss(recon, img, mu, log_var)
            eval_loss.append(loss.item())
    
    print('Test set: Average loss: {:.5f}\n'.format(np.mean(eval_loss)))
    return np.mean(eval_loss)

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

def test_3(checkpoint_dir, test_loader):
    model = VAE(args.latent_dim)
    load_checkpoint(checkpoint_dir, model)
    mseloss = nn.MSELoss()
    model.to(device)
    model.eval()

    latent = []
    with torch.no_grad():
        for batch_idx, img in enumerate(test_loader):
            img = img.to(device)
            if batch_idx < 10:
                mu, log_var, recon = model(img)

                loss, mse, kld = VAE_loss(mseloss, recon, img, mu, log_var)

                torchvision.utils.save_image(img, f'./hw3_data/vae_output2/{batch_idx}_ori.jpg')
                torchvision.utils.save_image(recon, f'./hw3_data/vae_output2/{batch_idx}_re.jpg')
                print(f'Saving {batch_idx} mse:{mse.item()}')
            x = model.encoder(img)
            x = torch.flatten(x, start_dim=1)
            mu = model.fc_mu(x)
            latent.append(mu.squeeze().cpu().numpy())
    tsne(latent)
    print("Done")

def test(checkpoint_dir, save_path):
    
    rand_variable = torch.randn(32, 512).to(device)

    model = VAE(args.latent_dim)
    load_checkpoint(checkpoint_dir, model)
    mseloss = nn.MSELoss()
    model.to(device)

    model.eval()

    recon = model.decode_input(rand_variable)
    recon = recon.view(-1, 512, 2, 2)
    recon = model.decoder(recon)
    recon = model.final_layer(recon)

    torchvision.utils.save_image(recon, os.path.join(save_path) ,nrow=8) # "./hw3_data/vae_generate/fig1_4_90_tmp.jpg"
    print("Done")

def tsne(latent):
    df = pd.read_csv('./hw3_data/face/test.csv')
    male_label = np.array(df['Male'])

    print('Start tsne')
    # pdb.set_trace()
    X_tsne = TSNE(n_components=2).fit_transform(latent)
    print('end tsne')

    plt.figure(figsize=(16,10))
    for i in [0,1]:
        if i==1:
            color="green"
            gender = "Male"
        else:
            color="red"
            gender = "Female"
        xy = X_tsne[male_label==i]
        plt.scatter(xy[:,0], xy[:,1], c=color, label=gender,
                alpha=0.3)
    plt.legend()
    plt.title("Gender")
    # plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y)
    plt.savefig('tsne_1.jpg')
    pdb.set_trace()


if __name__=='__main__':
    if args.test_generate_path:
        print("Start generating ...")
        test(args.checkpoint_dir, args.test_generate_path) 
    elif args.tsne:
        print("Start testing and tsne ...")
        test_dataset = FaceDataset(args.test_img_dir)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        test_3(args.checkpoint_dir, test_loader)
    else:
        print(f"Start training on {args.train_img_dir}")
        train_dataset = FaceDataset(args.train_img_dir)
        valid_dataset = FaceDataset(args.valid_img_dir)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        train(args.lr, args.batch_size, args.latent_dim, train_loader, valid_loader, args.restore)


