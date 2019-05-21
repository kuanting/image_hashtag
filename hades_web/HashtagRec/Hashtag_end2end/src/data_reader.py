import numpy as np
import pandas as pd
from scipy.spatial import distance
from PIL import Image, ImageFile
import scipy, argparse, os, sys, csv, io, time

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.img_extractor import img_centre_crop

class data_reader(Dataset):
    def __init__(self, folder_dir, word2idx, word_vec_dict, train=True):
        self.train = train
        self.img_centre_crop = img_centre_crop()

        self.folder_dir = folder_dir
        self.data = pd.read_csv(os.path.join(folder_dir+'data_list.txt'), header=None).values
        self.tag = pd.read_csv(os.path.join(folder_dir+'tag_list.txt'), header=None).values
        self.total_num = len(self.data)
        self.rdn_idx = np.arange(self.total_num)
        np.random.shuffle(self.rdn_idx)
        self.data = self.data[self.rdn_idx]
        self.tag = self.tag[self.rdn_idx]
        
        self.x = self.data[:-int(0.2*self.total_num)]
        self.y = self.tag[:-int(0.2*self.total_num)]
        self.vx = self.data[-int(0.2*self.total_num):]
        self.vy = self.tag[-int(0.2*self.total_num):]

        self.word2idx = word2idx
        self.word_vec_dict = word_vec_dict

    def __len__(self):
        return(len(self.x) if self.train else len(self.vx))
    def __getitem__(self, idx):
        tmp_x = self.x[idx] if self.train else self.vx[idx]
        tmp_y = self.y[idx] if self.train else self.vy[idx]
        hashtag = tmp_y[0].strip()
        category = tmp_x[0].split('/')[1]

        category = category[:-1] if category not in self.word2idx.keys() else category
        widx = self.word2idx[category]
        category_vec = self.word_vec_dict[widx]
        
        tmp_path = os.path.join(self.folder_dir ,tmp_x[0])
        img = Image.open(tmp_path).convert('RGB')

        input_img = self.img_centre_crop(img)

        return(input_img, (category_vec, hashtag))

if __name__ == "__main__":
    folder_dir = "../Hashtag/HARRISON/"
    # tr_img = data_reader(folder_dir,)
    # tr = DataLoader(tr_img, batch_size=64, shuffle=True)
    print("this is dataset")