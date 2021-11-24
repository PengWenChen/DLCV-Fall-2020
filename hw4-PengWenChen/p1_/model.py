import torch.nn as nn
import torch
import pdb

class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Parametric(nn.Module):
    def __init__(self, n_way, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
        self.relation = nn.Sequential(
            conv_block(hid_channels*2, hid_channels),
            conv_block(hid_channels, hid_channels),
            # conv_block(hid_channels, hid_channels),
            # conv_block(hid_channels, out_channels),
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.Linear(128, 32),
            nn.Linear(32, n_way),
            nn.Sigmoid(),
        )

    def forward(self, support, query):
        support = self.encoder(support)
        query = self.encoder(query)

        query = torch.unsqueeze(query, dim=1)
        query = query.repeat(1,5,1,1,1)
        query = query.reshape(75*5,64,5,5)

        support = support.repeat(75,1,1,1)

        dis = torch.cat((support, query), dim=1) # 375*64*5*5
        dis = self.relation(dis) # 375*64*1*1
        dis = dis.reshape(75,5,64,1,1)
        dis = dis.view(dis.size(0), -1)
        dis = self.fc(dis)
        return dis