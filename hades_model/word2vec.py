import numpy as np
import pandas as pd
import scipy, argparse, os, sys, csv, io, time

import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

### Path of data ###
folder_dir = "/home/chihen/NTU/Research/Hashtag/HARRISON/"
filename = "tag_list.txt"

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def command():
    parser = argparse.ArgumentParser(description='Choose the mode you want')
    parser.add_argument("-m", "--mode", type=str, dest="mode",
                        help="train or test, Default is train", default="train")
    parser.add_argument("-bs", "--batch_size", type=int, dest="bs",
                        help="batch_size of network, Default is 64", default=64)
    parser.add_argument("-epos", "--epochs", type=int, dest="epos",
                        help="Numbers of epochs, Default is 100", default=100)
    parser.add_argument("-o", "--output", type=str, dest="output",
                        help="Path of output file, default is ./", default="./")
    parser.add_argument("-lr", type=float, dest="lr",
                        help="Default is 0.001", default=0.001)
    parser.add_argument("-w", "--skip_window", type=int, dest="w",
                        help="skip-gram window size, default is 3", default=3)
    parser.add_argument("-th", "--threshold", type=int, dest="th",
                        help="threshold of word, default is 4", default=4)
    parser.add_argument("-embed", "--embedding_size", type=int, dest="embed",
                        help="Embedding size of word vector, default is 500", default=500)
    return parser

def read_tags(folder_dir, filename, args):
    read_data = []
    word_counter = {}
    file_path   = os.path.join(folder_dir, filename)
    with open(file_path, 'r', encoding="utf-8") as f:
        tmp = f.readlines()
        for content in tmp:
            content = content.strip('\n').strip() 
            test = content.strip().split()
            if len(test) > 1:
                read_data.append(test)
                for ele in test:
                    if ele not in word_counter.keys():
                        word_counter[ele] = 1
                    else:
                        word_counter[ele] += 1
    corpus = [k for k,v in word_counter.items() if v > args.th]
    word2idx = {k:v for v,k in enumerate(corpus, 1)}
    word2idx['UNK'] = 0
    idx2word = {v:k for v,k in enumerate(corpus, 1)}
    idx2word[0] = 'UNK'
    corpus = corpus+['UNK']
    return read_data, word_counter, corpus, word2idx, idx2word

def convert2pair(data, corpus, word2idx, args):
    idx_data = data.copy()
    ### data convert to idx ###
    for i, row in enumerate(idx_data):
        for i_ele, ele in enumerate(row):
            if ele not in word2idx.keys():
                idx_data[i][i_ele] = 0
            else:
                idx_data[i][i_ele] = word2idx[ele]
    
    training_set = []
    ### create training pair ###
    for i, row in enumerate(idx_data):
        for i_ele, ele in enumerate(row):
            for slide in range(-args.w, args.w+1):
                if i_ele+slide < 0 or i_ele+slide >= len(row) or slide == 0:
                    continue
                else:
                    training_set.append([ele, row[i_ele+slide]])
    return training_set

class word2vec_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return(len(self.data))
    def __getitem__(self, idx):
        return (self.data[idx][0], self.data[idx][1])

class word2vec(nn.Module):
    def __init__(self, args, vol_size):
        super(word2vec, self).__init__()
        self.embed = args.embed
        self.vol = vol_size
        self.u_embeddings = nn.Embedding(self.vol, self.embed)
        self.linear = nn.Linear(self.embed, self.vol)
        self.act = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        embed_vec = self.u_embeddings(x)
        x = self.linear(embed_vec)
        x = self.act(x)
        return x, embed_vec

if __name__ == "__main__":
    args = command().parse_args()
    data, word_counter, corpus, word2idx, idx2word = read_tags(folder_dir, filename, args)

    wordpair = convert2pair(data, corpus, word2idx, args)
    training_data = DataLoader(word2vec_dataset(wordpair), batch_size=args.bs, shuffle=True)
    
    word2vec_model = word2vec(args, len(corpus)).to(device)
    optimizer = optim.Adam(word2vec_model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    
    for epoch in range(args.epos):
        total_loss = 0
        print("This is epo: {:4d}".format(epoch))
        est = time.time()
        word2vec_model.train()
        for i,(x,y) in enumerate(training_data, 1):
            x = x.to(device)
            y = y.to(device)
            word2vec_model.zero_grad()
            y_pred, _ = word2vec_model(x)
            loss = criterion(y_pred.view(-1, len(corpus)), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print("Batch: {:6d}/{:6d} loss: {:06.4f}".format(i, len(training_data), total_loss/i),
                  end = '\r' if i != len(training_data) else '\n')

        print("time: {:04.2f}s".format(time.time()-est))
        print("--- This is test part ---")
        word2vec_model.eval()
        with torch.no_grad():
            total = np.arange(len(corpus)).reshape((len(corpus),1)) 
            tx = np.random.randint(len(corpus), size=1)
            torch_total = torch.from_numpy(total).to(device)
            _, embed_total = word2vec_model(torch_total)
            embed_total = embed_total.reshape((len(corpus), -1))
            
            embed_total_numpy = embed_total.cpu().numpy()

            ### strange cosine ###
            dists = embed_total_numpy[tx[0]].dot(embed_total_numpy.transpose())
            knn_word_ind = (-dists).argsort()[0:10]
            ######################

            print("Top@10 close to {} are: ".format(idx2word[tx[0]]), end=" ")
            for i in knn_word_ind:
                print(idx2word[i], end=" ")
            print()
    print("Saving the word_embedding vector...")
    with torch.no_grad():
        idx2wordvec = {}
        total = torch.from_numpy(np.arange(len(corpus)).reshape(-1,1)).to(device)
        _, embed_total = word2vec_model(total)
        embed_total = embed_total.reshape((len(corpus), -1))
        embed_total_numpy = embed_total.cpu().numpy()
        output_path = os.path.join(args.output, 'wordvec.npz')
        # idx2wordvec = {idx:row for idx, row in enumerate(embed_total_numpy)}
        np.savez(output_path, wordvec=embed_total_numpy)
         
