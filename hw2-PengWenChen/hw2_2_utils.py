from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import random
from PIL import Image
import PIL
import os
import pdb

class SegDataset(Dataset):
    def __init__(self, root, train=False):
        self.root = root
        self.data_path = []

        self.train = train

        for img_file in sorted(os.listdir(self.root)):
            if img_file.split('.')[0].split('_')[1]=='sat':
                self.data_path.append(img_file)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, item):
        path = os.path.join(self.root, self.data_path[item])
        img = Image.open(path)
        t_path = os.path.join(self.root, self.data_path[item].split('_')[0]+'_mask.png')
        target = Image.open(t_path)

        if self.train:
            if random.random() > 0.5:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                target = target.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:
                img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                target = target.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        img = self.transform(img)
        target = np.array(target)
        target_map = self.read_masks(target, target.shape).astype(np.uint8)

        # img, target = self.transform(img, target_map)
        target = torch.tensor(target_map)
        return img, target
    
    def read_masks(self, seg, shape):
        masks = np.zeros((shape[0], shape[1]))
        mask = (seg >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown
        masks[mask == 4] = 6  # (Red: 100) Unknown
        # pdb.set_trace()
        return masks

    def __len__(self):
        return len(self.data_path)

class SegDataset_for_test(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_path = []

        for img_file in sorted(os.listdir(self.root)):
            if img_file.split('.')[0].split('_')[1]=='sat':
                self.data_path.append(img_file)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, item):
        path = os.path.join(self.root, self.data_path[item])
        img = Image.open(path)
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data_path)

class Vggfcn32(nn.Module):
    def __init__(self, pretrain=False):
        super(Vggfcn32, self).__init__()
        self.pretrain = pretrain
        self.layer1 = pretrain.features
        self.relu = nn.ReLU(inplace=True)
        self.dev1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dev2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dev3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dev4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dev5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 7, 1)

    def forward(self, input):
        x = input
        x = self.layer1(x)
        x = self.bn1(self.relu(self.dev1(x)))
        x = self.bn2(self.relu(self.dev2(x)))
        x = self.bn3(self.relu(self.dev3(x)))
        x = self.bn4(self.relu(self.dev4(x)))
        x = self.bn5(self.relu(self.dev5(x)))
        x = self.conv7(x)
        # pdb.set_trace()
        # x = self.convTranspose(x)
        # x = F.upsample_bilinear(x, input.size()[2:])
        return x

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class Vggfcn8(nn.Module):
    def __init__(self, pretrain=False):
        super(Vggfcn8, self).__init__()
        print("It's Vggfcn8_bn")
        self.pretrain = pretrain
        self.pretrain_all = pretrain.features

        self.block1 = pretrain.features[0:7] #[0:5]
        self.block2 = pretrain.features[7:14] #[5:10]
        self.block3 = pretrain.features[14:24] #[10:17]
        self.block4 = pretrain.features[24:34] #[17:24]
        self.block5 = pretrain.features[34:44] #[24:31]

        self.relu = nn.ReLU(inplace=True)
        self.dev1 = nn.ConvTranspose2d(512, 512, kernel_size=3, 
                    stride=2, padding=1, dilation=1, output_padding=1)
        self.dev1_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, 
                    stride=2, padding=1, dilation=1, output_padding=1)
        self.dev2 = nn.ConvTranspose2d(256, 256, kernel_size=3, 
                    stride=2, padding=1, dilation=1, output_padding=1)
        self.dev3 = nn.ConvTranspose2d(128, 128, kernel_size=3, 
                    stride=2, padding=1, dilation=1, output_padding=1)
        self.dev4 = nn.ConvTranspose2d(64, 64, kernel_size=3, 
                    stride=2, padding=1, dilation=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 7, 1)

        self.conv1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 7, 3, stride=1, padding=1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, input):
        x = input

        x1 = self.block1(x) # 64, 256, 256
        x2 = self.block2(x1) # 128, 128, 128
        x3 = self.block3(x2) # 256, 64, 64
        x4 = self.block4(x3) # 512, 32, 32
        x5 = self.block5(x4) # 512, 16, 16

        x5 = self.conv1(x5)
        u1 = self.relu(self.bn1(self.dev1(x5))) # 512, 32, 32

        u2 = self.relu(self.bn1_1(self.dev1_1(u1+x4))) # 256, 64, 64
        u2 = self.conv2(u2) # 256, 32, 32

        u3 = self.relu(self.bn2(self.dev2(u2+x3))) # 128, 128, 128
        u3 = self.conv3(u3) # 256, 64, 64

        u4 = self.relu(self.bn3(self.dev3(u3+x2))) # 64, 256, 256
        u4 = self.conv4(u4) # 128, 128, 128

        u5 = self.relu(self.bn4(self.dev4(u4+x1))) # 32, 512, 512
        u5 = self.conv5(u5) # 64, 256, 256

        output = self.conv6(u5) 
        # pdb.set_trace()
        return output


cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],
}

def label2RGB(output):
    # pdb.set_trace()
    output = np.stack((output,)*3, axis=-1)
    for i in range(6):
        if (output==i).any():
            location = np.where(output == i)
            try:
                output[location[0], location[1], :]=cls_color[i]
            except:
                pdb.set_trace()
    return output

def save(output, output_path, index=0):
    print("Saving %04d_mask.png" % index, end='\r')
    try:
        im = Image.fromarray(output.astype(np.uint8))
        path = os.path.join(output_path, "%04d_mask.png" % index)
        im.save(path)
    except:
        pdb.set_trace()


class Vggfcn32_simple(nn.Module):
    def __init__(self, pretrain=False):
        super(Vggfcn32, self).__init__()
        self.pretrain = pretrain
        self.layer1 = pretrain.features
        self.relu = nn.ReLU(inplace=True)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.up = nn.ConvTranspose2d(32, 7, 32, stride=32)

    def forward(self, input):
        x = input
        x = self.layer1(x)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.up(x)

        # x = self.convTranspose(x)
        # x = F.upsample_bilinear(x, input.size()[2:])
        return x




