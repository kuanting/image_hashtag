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

class heracles_data_reader(Dataset):
    def __init__(self, folder_dir, word2idx, word_vec_dict, train=True, model_type='heracles'):
        ### Use C.E data ###
        self.data = np.load('data/image_feature_vgg19bn_tr.npz') if train else np.load('data/image_feature_vgg19bn_te.npz')
        self.x = self.data['image_feature']
        self.y_cate = self.data['total_label']
        self.y_hashtag = self.data['total_hashtag']
        ####################

        self.model_type = model_type
        self.word2idx = word2idx
        self.word_vec_dict = word_vec_dict

    def __len__(self):
        return(len(self.x))

    def __getitem__(self, idx):
        tmp_x = self.x[idx]
        tmp_y_hashtag = self.y_hashtag[idx]
        tmp_y_cate = self.y_cate[idx]

        # remove suffix (s)
        tmp_y_cate = tmp_y_cate[:-1] if tmp_y_cate not in self.word2idx.keys() else tmp_y_cate 
        
        # # find better way, now is word->idx->vec
        # widx = self.word2idx[tmp_y_cate]
        # category_vec = self.word_vec_dict[widx]
        
        ### New category for devise, mean of hashtagvec ###
        use_to_mean_list = []
        tmp_y_hashtag_split = tmp_y_hashtag.split()
        prob = 1/(len(tmp_y_hashtag_split)+1)
        for ele in tmp_y_hashtag_split:
            widx = self.word2idx[ele] if ele in self.word2idx.keys() else self.word2idx['UNK']
            tmp_category_vec = self.word_vec_dict[widx]*prob
            tmp_category_vec = tmp_category_vec*2 if self.word2idx[tmp_y_cate] == widx else tmp_category_vec
            use_to_mean_list.append(tmp_category_vec)
        category_vec = np.sum(np.array(use_to_mean_list),axis=0)
        ###################################################

        return(tmp_x, (category_vec, tmp_y_hashtag))

if __name__ == "__main__":
    folder_dir = "../Hashtag/HARRISON/"
    print("this is dataset")
    print(len(word2idx))
