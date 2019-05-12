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

def cos_cdist(matrix, v):
   # v = v.reshape(1,-1)
   # return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
   return scipy.spatial.distance.cdist(matrix, v, 'cosine').T

def top_k(y_pred, vec_matrix, k):
    # total_top_k = []
    # for i in range(len(y_pred)):
    #     tmp_get = cos_cdist(vec_matrix, y_pred[i])
    #     tmp_k = np.argsort(tmp_get)[:k]
    #     total_top_k.append(tmp_k)
    # return np.array(total_top_k)
    tmp = cos_cdist(vec_matrix, y_pred)
    total_top_k = np.argsort(tmp)[:, :k]
    return total_top_k

def f1_score(top_k_result, hashtag_y, idx2word):
    gt_hashtag, to_word = [], []
    tp, total_l = 0, 0
    for ele in hashtag_y:
        ele = ele.strip().split()
        gt_hashtag.append(ele)
    for row in range(len(top_k_result)):
        tmp = []
        for ele in top_k_result[row]:
            tmp += [idx2word[ele]]
        to_word.append(tmp)
    for row_idx in range(len(to_word)):
        tp += len(set(to_word[row_idx])&set(gt_hashtag[row_idx]))
        total_l += len(gt_hashtag[row_idx])
    return tp, total_l

if __name__ == "__main__":
    print("this is evalution pyfile")