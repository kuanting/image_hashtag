import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy, argparse, os, sys, csv, io, time

import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from word2vec import read_tags

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def command():
    parser = argparse.ArgumentParser(description='Choose the mode you want')
    parser.add_argument("-m", "--mode", type=str, dest="mode", help="train or test, Default is train", default="train")
    parser.add_argument("-bs", "--batch_size", type=int, dest="bs", help="batch_size of network, Default is 64", default=64)
    parser.add_argument("-epos", "--epochs", type=int, dest="epos", help="Numbers of epochs, Default is 100", default=100)
    parser.add_argument("-o", "--output", type=str, dest="output", help="Path of output file, default is output/", default="output/")
    parser.add_argument("-lr", type=float, dest="lr", help="Default is 0.001", default=0.001)
    parser.add_argument("-w", "--skip_window", type=int, dest="w", help="skip-gram window size, default is 3", default=3)
    parser.add_argument("-th", "--threshold", type=int, dest="th", help="threshold of word, default is 4", default=4)
    parser.add_argument("-embed", "--embedding_size", type=int, dest="embed", help="Embedding size of word vector, default is 500", default=500)
    return parser

class img2vec(nn.Module):
    def __init__(self):
        super(img2vec, self).__init__()
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 500) 
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class img_feature(Dataset):
    def __init__(self, image_data, word2idx, word_vec_dict, train=True):
        self.train = train
        self.data = np.load(image_data)
        self.word2idx = word2idx
        self.word_vec_dict = word_vec_dict
        self.x = self.data['image_feature']
        self.y = self.data['total_label']
        self.tx = self.data['val_image_feature']
        self.ty = self.data['val_total_label']
        self.hashtag = self.data['total_hashtag']
        self.thashtag = self.data['val_total_hashtag']
    def __len__(self):
        return(len(self.x) if self.train else len(self.tx))
    def __getitem__(self,idx):
        tmp_y = self.y[idx] if self.train else ty[idx]
        tmp_hashtag = self.hashtag[idx] if self.train else self.thashtag[idx]
        # tmp_hashtag = tmp_hashtag.strip().split()
        # for idx, ele in enumerate(tmp_hashtag):
            # widx = self.word2idx[ele] if ele in self.word2idx.keys() else self.word2idx['UNK']
            # tmp_hashtag[idx] = widx
            # tmp_hashtag[idx] = self.word_vec_dict[widx]
        if tmp_y not in word2idx.keys():
            tmp_y = tmp_y[:-1]
        widx = self.word2idx[tmp_y] 
        tmp_y = self.word_vec_dict[widx]
        return(self.x[idx] if self.train else self.tx[idx], (tmp_y, tmp_hashtag))

def cos_cdist(matrix, v):
   v = v.reshape(1,-1)
   return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

def top_k(y_pred, vec_matrix, k):
    total_top_k = []
    for i in range(len(y_pred)):
        tmp_get = cos_cdist(vec_matrix, y_pred[i])
        tmp_k = np.argsort(tmp_get)[:k]
        total_top_k.append(tmp_k)
    return np.array(total_top_k)

def f1_score(top_k_result, hashtag_y, idx2word):
    gt_hashtag, to_word = [], []
    tp, total_l = 0, 0
    for ele in hashtag_y:
        ele = ele.strip().split()
        gt_hashtag.append(ele)
    for row in range(len(top_k_result)):
        tmp = []
        for ele in top_k_result[row]:
            # tmp += idx2word[ele] + ' '
            tmp += [idx2word[ele]]
        # to_word.append(tmp.strip())
        to_word.append(tmp)
    for row_idx in range(len(to_word)):
        tp += len(set(to_word[row_idx])&set(gt_hashtag[row_idx]))
        total_l += len(gt_hashtag[row_idx])
    return tp, total_l

if __name__ == "__main__":
    folder_dir = "../Hashtag/HARRISON/"
    filename = "tag_list.txt"
    vec_matrix = np.load('wordvec.npz')['wordvec']
    print("Total wordvec in dataset: ", vec_matrix.shape, " <num_word, dim>")
    word_vec_dict = {idx:vec for idx,vec in enumerate(vec_matrix)}

    args = command().parse_args()

    _, _, corpus, word2idx, idx2word = read_tags(folder_dir, filename, args)

    tr_img = img_feature('image_feature.npz', word2idx, word_vec_dict)
    te_img = img_feature('image_feature.npz', word2idx, word_vec_dict, train=False)

    tr = DataLoader(tr_img, batch_size=64, shuffle=True)
    te = DataLoader(te_img, batch_size=64, shuffle=True)

    model = img2vec().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(200):
        print('This is epoch {:4d}'.format(epoch))
        total_loss, all_tp, all_total = 0, 0, 0
        st = time.time()
        for i,(x,y) in enumerate(tr, 1):
            cate_y, hashtag_y = y
            x = x.to(device)
            # y = y.to(device)
            cate_y = cate_y.to(device)
            model.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, cate_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            ### find top@k of y_pred and evalution ###
            y_pred_find = y_pred.detach().cpu()
            top_k_result = top_k(y_pred_find, vec_matrix, 10)
            tp, total_l = f1_score(top_k_result, hashtag_y, idx2word)
            all_tp += tp
            all_total += total_l
            recall_rate = all_tp/all_total *100 
            print("Batch: {:4d}/{:4d} loss: {:06.4f}, recall: {:04.2f}%".format(i, len(tr), total_loss/i, recall_rate),
                  end=' '*5+'\r' if i != len(tr) else ' ')
        print('total time: {:>5.2f}s'.format(time.time()-st))
         
