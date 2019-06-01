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

from src.image_model import img_centre_crop

class data_reader(Dataset):
    def __init__(self, folder_dir, word2idx, word_vec_dict, train=True, model_type='devise'):
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

        self.model_type = model_type

    def __len__(self):
        return(len(self.x) if self.train else len(self.vx))

    def __getitem__(self, idx):
        tmp_x = self.x[idx] if self.train else self.vx[idx]
        tmp_y = self.y[idx] if self.train else self.vy[idx]
        hashtag = tmp_y[0].strip()
        category = tmp_x[0].split('/')[1]
        
        # remove suffix (s)
        category = category[:-1] if category not in self.word2idx.keys() else category
        
        # # find better way
        # widx = self.word2idx[category]
        # category_vec = self.word_vec_dict[widx]
        
        ### New category for devise, mean of hashtagvec ###
        use_to_mean_list = []
        tmp_y_hashtag_split = hashtag.split()
        prob = 1/(len(tmp_y_hashtag_split)+1)
        for ele in tmp_y_hashtag_split:
            widx = self.word2idx[ele] if ele in self.word2idx.keys() else self.word2idx['UNK']
            tmp_category_vec = self.word_vec_dict[widx]*prob
            tmp_category_vec = tmp_category_vec*2 if self.word2idx[category] == widx else tmp_category_vec
            use_to_mean_list.append(tmp_category_vec)
        category_vec = np.sum(np.array(use_to_mean_list),axis=0)
        ###################################################

        tmp_path = os.path.join(self.folder_dir ,tmp_x[0])
        img = Image.open(tmp_path).convert('RGB')

        input_img = self.img_centre_crop(img)
        if self.model_type == 'devise' or self.model_type == 'convex':
            return(input_img, (category_vec, hashtag))
        elif self.model_type == 'dnn':
            return(input_img, (widx, hashtag))
        elif self.model_type == 'multi_label':
            hashtag = hashtag.strip()
            split_tag = hashtag.split()
            hashtag_idx = []
            one_hot_hashtag = np.zeros((len(self.word2idx),))
            for ele in split_tag:
                ele = 0 if ele not in self.word2idx.keys() else self.word2idx[ele]
                hashtag_idx.append(ele)
            one_hot_hashtag[hashtag_idx] = 1
            return(input_img, (one_hot_hashtag, hashtag))

if __name__ == "__main__":
    folder_dir = "../Hashtag/HARRISON/"
    # tr_img = data_reader(folder_dir,)
    # tr = DataLoader(tr_img, batch_size=64, shuffle=True)
    print("this is dataset")
    print(len(word2idx))
