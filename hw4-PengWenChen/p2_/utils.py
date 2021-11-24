from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CosineSimilarity
import pdb
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def cosine_sim(a1, b1): #a1:(75, 1600) b1:(5, 1600)

    n = a1.shape[0]
    m = b1.shape[0]
    a1 = a1.unsqueeze(1).expand(n, m, -1)
    b1 = b1.unsqueeze(0).expand(n, m, -1)

    cos = CosineSimilarity(dim=2, eps=1e-6)
    output = cos(a1, b1)
    return output
# def parametric(a, b):
    


