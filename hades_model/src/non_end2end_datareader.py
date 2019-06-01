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

class data_reader(Dataset):
    def __init__(self, folder_dir, word2idx, word_vec_dict, train=True, model_type='devise'):
        ### Use C.E data ###
        self.data = np.load('data/image_feature_resnet50_tr.npz') if train else np.load('data/image_feature_resnet50_te.npz')
        self.x = self.data['image_feature']
        self.y_cate = self.data['total_label']
        self.y_hashtag = self.data['total_hashtag']
        ####################

        ### Use K.T data ###
        # self.data = np.load('data/harrison_features.npz')
        # self.idxx = self.data['train_indices'] if train else self.data['test_indices']
        # self.x = self.data['imagenet_fc_layers'][self.idxx]
        # self.y_hashtag = self.data['hashtag_list'][self.idxx]
        # self.y_cate = self.data['image_list'][self.idxx]
        ####################

        self.model_type = model_type
        self.word2idx = word2idx
        self.word_vec_dict = word_vec_dict

    def __len__(self):
        return(len(self.x))

    def __getitem__(self, idx):
        tmp_x = self.x[idx][np.newaxis, :]
        tmp_y_hashtag = self.y_hashtag[idx]
        tmp_y_cate = self.y_cate[idx]

        ### Use K.T data ###
        # tmp_y_cate = tmp_y_cate.split('/')[1]
        ####################

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

        if self.model_type == 'devise' or self.model_type == 'convex':
            return(tmp_x, (category_vec, tmp_y_hashtag))
        elif self.model_type == 'dnn':
            return(tmp_x, (widx, tmp_y_hashtag))
        elif self.model_type == 'multi_label':
            tmp_y_hashtag = tmp_y_hashtag.strip()
            split_tag = tmp_y_hashtag.split()
            hashtag_idx = []
            one_hot_hashtag = np.zeros((len(self.word2idx),))
            for ele in split_tag:
                ele = 0 if ele not in self.word2idx.keys() else self.word2idx[ele]
                hashtag_idx.append(ele)
            one_hot_hashtag[hashtag_idx] = 1
            return(tmp_x, (one_hot_hashtag, tmp_y_hashtag))

if __name__ == "__main__":
    folder_dir = "../Hashtag/HARRISON/"
    print("this is dataset")
    print(len(word2idx))
