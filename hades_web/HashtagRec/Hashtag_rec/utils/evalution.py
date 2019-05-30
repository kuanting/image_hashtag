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

class log_manager_heracles():
    def __init__(self, args):
        self.G = np.zeros((2,))
        self.D = np.zeros((2,))
        self.total_tp = np.zeros((2,))
        self.total_len = np.zeros((2,))
        self.Dx = np.zeros((2,))
        self.Dx_score = np.zeros((2,))
        self.D_Gz1  = np.zeros((2,))
        self.D_Gz1_score = np.zeros((2,))
        self.D_Gz2 = np.zeros((2,))
        self.D_Gz2_score = np.zeros((2,))
        self.loss_Dx = np.zeros((2,))
        self.loss_D_Gz1 = np.zeros((2,))
        self.loss_D_Gz2 = np.zeros((2,))
        self.recall = np.zeros((2,))
        self.tr_total_loss_Dx = []
        self.tr_total_loss_D_Gz1 = []
        self.tr_total_loss_D_Gz2 = []
        self.te_total_loss_Dx = []
        self.te_total_loss_D_Gz1 = []
        self.te_total_loss_D_Gz2 = []
        self.tr_recall = []
        self.te_recall = []
    def accumlate_D(self, Dx, Dx_score, D_Gz1, D_Gz1_score, train=True):
        mode = 0 if train else 1
        self.D[mode] += 1
        self.Dx[mode] += Dx
        self.Dx_score[mode] += Dx_score
        self.D_Gz1[mode] += D_Gz1
        self.D_Gz1_score[mode] += D_Gz1_score
        self.loss_Dx[mode] += (2 * self.Dx[mode] + 3 * self.Dx_score[mode])
        self.loss_D_Gz1[mode] += (2 * self.D_Gz1[mode] + 3 * self.D_Gz1_score[mode])
    def accumlate_G(self, D_Gz2, D_Gz2_score, train=True):
        mode = 0 if train else 1
        self.G[mode] += 1
        self.D_Gz2[mode] += D_Gz2
        self.D_Gz2_score[mode] += D_Gz2_score
        self.loss_D_Gz2[mode] += (2 * self.D_Gz2[mode] + 3 * self.D_Gz2_score[mode])
    def accumlate_R(self, tp, total_l, train=True):
        mode = 0 if train else 1
        self.total_tp[mode] += tp
        self.total_len[mode] += total_l
        self.recall[mode] = self.total_tp[mode] / self.total_len[mode] *100
    def display(self, train=True):
        mode = 0 if train else 1
        return (self.Dx[mode]/self.D[mode], self.D_Gz1[mode]/self.D[mode],
                self.D_Gz2[mode]/self.G[mode], self.recall[mode])
    def record_log(self):
        self.tr_total_loss_Dx.append(self.Dx[0]/self.D[0])
        self.tr_total_loss_D_Gz1.append(self.D_Gz1[0]/self.D[0])
        self.tr_total_loss_D_Gz2.append(self.D_Gz2[0]/self.G[0])
        self.te_total_loss_Dx.append(self.Dx[1]/self.D[1])
        self.te_total_loss_D_Gz1.append(self.D_Gz1[1]/self.D[1])
        self.te_total_loss_D_Gz2.append(self.D_Gz2[1]/self.G[1])
        self.tr_recall.append(self.recall[0])
        self.te_recall.append(self.recall[1])
        np.savez(os.path.join('log', 'log_{}.npz'.format('heracles')),
                 train_Dx= self.tr_total_loss_Dx,
                 train_D_Gz1 = self.tr_total_loss_D_Gz1,
                 train_D_Gz2 = self.tr_total_loss_D_Gz2,
                 train_recall = self.tr_recall,
                 test_Dx = self.te_total_loss_Dx,
                 test_D_Gz1 = self.te_total_loss_D_Gz1,
                 test_D_Gz2 = self.te_total_loss_D_Gz2,
                 test_recall = self.te_recall)
        self.clean_record()
    def clean_record(self):
        self.G = np.zeros((2,))
        self.D = np.zeros((2,))
        self.total_tp = np.zeros((2,))
        self.total_len = np.zeros((2,))
        self.Dx = np.zeros((2,))
        self.Dx_score = np.zeros((2,))
        self.D_Gz1  = np.zeros((2,))
        self.D_Gz1_score = np.zeros((2,))
        self.D_Gz2 = np.zeros((2,))
        self.D_Gz2_score = np.zeros((2,))
        self.loss_Dx = np.zeros((2,))
        self.loss_D_Gz1 = np.zeros((2,))
        self.loss_D_Gz2 = np.zeros((2,))
        self.recall = np.zeros((2,))
    def plot_log(self):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        plt.figure(figsize=(16,12))
        plt.clf()
        plt.subplot(2, 1, 1)
        l1 = np.array(self.tr_recall).flatten()
        l2 = np.array(self.te_recall).flatten()
        plt.xlabel('Epochs')
        plt.plot(l1, label='train')
        plt.plot(l2, label='test')
        plt.legend(loc='lower right')
        plt.title('{}_recall'.format('heracles'))
        plt.subplot(2, 3, 4)
        l1 = np.array(self.tr_total_loss_Dx).flatten()
        l2 = np.array(self.te_total_loss_Dx).flatten()
        plt.xlabel('Epochs')
        plt.plot(l1, label='train')
        plt.plot(l2, label='test')
        plt.legend(loc='upper right')
        plt.title('{}_D(x)'.format('heracles'))
        plt.subplot(2, 3, 5)
        l1 = np.array(self.tr_total_loss_D_Gz1).flatten()
        l2 = np.array(self.te_total_loss_D_Gz1).flatten()
        plt.xlabel('Epochs')
        plt.plot(l1, label='train')
        plt.plot(l2, label='test')
        plt.legend(loc='upper right')
        plt.title('{}_D(G(z1))'.format('heracles'))
        plt.subplot(2, 3, 6)
        l1 = np.array(self.tr_total_loss_D_Gz2).flatten()
        l2 = np.array(self.te_total_loss_D_Gz2).flatten()
        plt.xlabel('Epochs')
        plt.plot(l1, label='train')
        plt.plot(l2, label='test')
        plt.legend(loc='upper right')
        plt.title('{}_D(G(z2))'.format('heracles'))
        plt.tight_layout()
        plt.savefig(os.path.join('plot', 'heracles'))
        plt.close()

class log_manager():
    def __init__(self, args):
        self.model = args.mt
        self.end2end = "End-to-End" if args.end2end == "True" else "Non-End-to-End"
        self.count = np.zeros((2,))
        self.total_loss = np.zeros((2,))
        self.total_tp = np.zeros((2,))
        self.total_len = np.zeros((2,))
        self.recall = np.zeros((2,))
        self.loss = np.zeros((2,))

        self.tr_loss = []
        self.te_loss = []
        self.tr_recall = []
        self.te_recall = []
    def accumulate(self, loss, tp, total, train=True):
        mode = 0 if train else 1
        self.count[mode] += 1
        self.total_loss[mode] += loss
        self.total_tp[mode] += tp
        self.total_len[mode] += total
        self.recall[mode] = self.total_tp[mode] / self.total_len[mode] *100
        self.loss[mode] = self.total_loss[mode]/ self.count[mode]
    def record_log(self):
        self.tr_loss.append(self.loss[0])
        self.tr_recall.append(self.recall[0])
        self.te_loss.append(self.loss[1])
        self.te_recall.append(self.recall[1])
        np.savez(os.path.join('log', 'log_{}_{}.npz'.format(self.model, self.end2end)),
                 train_loss= self.tr_loss,
                 train_recall = self.tr_recall,
                 test_loss = self.te_loss,
                 test_recall = self.te_recall)
        self.clean_record()
    def clean_record(self):
        self.count = np.zeros((2,))
        self.total_loss = np.zeros((2,))
        self.total_tp = np.zeros((2,))
        self.total_len = np.zeros((2,))
        self.recall = np.zeros((2,))
        self.loss = np.zeros((2,))
    def plot_log(self):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        plt.figure(figsize=(16,12))
        plt.clf()
        plt.subplot(2, 1, 1)
        l1 = np.array(self.tr_loss).flatten()
        l2 = np.array(self.te_loss).flatten()
        plt.xlabel('Epochs')
        plt.plot(l1, label='train')
        plt.plot(l2, label='test')
        plt.legend(loc='upper right')
        plt.title('{}_{}_loss'.format(self.end2end, self.model))
        plt.subplot(2,1,2)
        l1 = np.array(self.tr_recall).flatten()
        l2 = np.array(self.te_recall).flatten()
        plt.xlabel('Epochs')
        plt.plot(l1, label='train')
        plt.plot(l2, label='test')
        plt.legend(loc='lower right')
        plt.title('{}_{}_recall'.format(self.end2end, self.model))
        plt.tight_layout()
        plt.savefig(os.path.join('plot', self.end2end+'_'+self.model))
        plt.close()

def cos_cdist(y_pred, vec_matrix):
   return scipy.spatial.distance.cdist(y_pred, vec_matrix, 'euclidean')

def top_k(y_pred, vec_matrix, k):
    tmp = cos_cdist(y_pred, vec_matrix)
    total_top_k = np.argsort(tmp)[:, :k]
    return total_top_k

def f1_score(top_k_result, hashtag_y, idx2word):
    # string compare to string
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
    a = np.array([[1,0,0], [0,1,0]])
    b = np.array([[1,0,0],[0,1,1]])
    c = np.array([[0,1,0]])
    print(a, b, c)
    print(cos_cdist(a,b))
    print(cos_cdist(a,c))
    print(np.argsort(cos_cdist(a,b)))
    print("this is evalution pyfile")
