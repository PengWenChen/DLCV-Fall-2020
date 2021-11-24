import torch.nn as nn
import torch
import copy
import numpy as np
import pdb

# For single training and single validation/testing
class ModelSingle(nn.Module):
    def __init__(self, pretrain):
        super().__init__()
        self.pretrain = pretrain

        self.fc_1 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 5),
        )

    def forward(self, input):
        feature = self.pretrain(input) # 1,1000
        feature = self.fc_1(feature)
        return feature

# For Sequence data and ensemble on testing. It wiill be faster.
# Dataloader will return seq data. And the model bellow will predict a seq output of a patient.
class ModelMultiEnsemble(nn.Module):
    def __init__(self, pretrain):
        super().__init__()
        self.pretrain = copy.deepcopy(pretrain)

        self.fc_1 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 5),
            #nn.Sigmoid(),
        )
        
    def forward(self, input, weight):
        # feature = input.unsqueeze(1).reshape(-1, 5, 3, 512, 512) # 5 image here
        # feature = feature.view(-1, 3, 512, 512)
        num = input.shape[0]
        feature = self.pretrain(input)
        feature = self.fc_1(feature)
        head = feature[0].unsqueeze(0)
        tail = feature[-1].unsqueeze(0)
        pad_feature = torch.cat([head, head, feature, tail, tail], dim=0)

        out = feature * 0.5
        out += pad_feature[0:num]*0.1
        out += pad_feature[1:num+1]*0.3
        out += pad_feature[3:num+3]*0.3
        out += pad_feature[4:num+4]*0.1
        # output = feature*weight
        # output = output.sum(1)

        return out

# Lewis 
# Use for fineturne or validation on multi image
class ModelMulti(nn.Module):
    def __init__(self, pretrain):
        super().__init__()
        self.pretrain = copy.deepcopy(pretrain)

        self.fc_1 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 5),
        )
        
    def forward(self, input, weight):
        feature = input.unsqueeze(1).reshape(-1, 5, 3, 512, 512) # 5 image here
        feature = feature.view(-1, 3, 512, 512)
        feature = self.pretrain(feature)
        feature = self.fc_1(feature)
        feature = feature.view(-1, 5, 5)
        output = feature*weight
        output = output.sum(1)
        # pdb.set_trace()
        
        #input: [batch, 9, 512, 512]
        # img0=input[:,0:3,:,:]
        # img1=input[:,3:6,:,:]
        # img2=input[:,6:9,:,:]
        
        # feature0 = self.pretrain(img0) # 1,1000
        # feature0 = self.fc_1(feature0)

        # feature1 = self.pretrain(img1) # 1,1000
        # feature1 = self.fc_1(feature1)

        # feature2 = self.pretrain(img2) # 1,1000
        # feature2 = self.fc_1(feature2)

        # output=feature0+feature1+feature2

        return output
