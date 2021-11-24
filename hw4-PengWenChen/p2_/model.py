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
        self.hallucination = nn.Sequential(
            nn.Linear(1600, 1600),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1600, 1600),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1600, 1600),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.MLP = nn.Sequential(
            nn.Linear(1600, 1600),
            nn.ReLU(),
        )

    def forward(self, x, M=1, noise=None, query=False, halluci=False):
        x = self.encoder(x)
        ori_data = x.view(x.size(0), -1) # (10,1600)
        if halluci:
            repeat_data = ori_data
            data = repeat_data + noise
            aug_data = self.hallucination(data)
            output = self.MLP(aug_data)
            return output
        if not query:
            repeat_data = ori_data.repeat(M,1)
            data = repeat_data + noise
            aug_data = self.hallucination(data)
            output = torch.cat((ori_data, aug_data), dim=0)
            output = self.MLP(output)
        else:
            output = self.MLP(ori_data)
        return output

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

