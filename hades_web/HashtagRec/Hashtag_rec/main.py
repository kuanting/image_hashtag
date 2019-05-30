import numpy as np
import pandas as pd
from scipy.spatial import distance
from PIL import Image, ImageFile
import scipy, argparse, os, sys, csv, io, time

import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset 

from src.image_model import img_centre_crop, img_model
from src.non_end2end_datareader import data_reader
from src.devise import devise
from src.multi_label import multi_label
from src.dnn import dnn
from src.heracles import Heracles, Labours
from src.train import training
from src.heracles_train import train_heracles
from utils.evalution import top_k, f1_score, cos_cdist
from word2vec import read_tags

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def command():
    parser = argparse.ArgumentParser(description='Choose the mode you want')
    parser.add_argument("-m", "--mode", type=str, dest="mode",
                        help="train or test, Default is train", default="train", choices=['train', 'test'])
    parser.add_argument("-bs", "--batch_size", type=int, dest="bs",
                        help="Batch_size of network, Default is 64", default=64)
    parser.add_argument("-epos", "--epochs", type=int, dest="epos",
                        help="Numbers of epochs, Default is 200", default=100)
    parser.add_argument("-s", "--save", type=str, dest="save",
                        help="Path of training model, default is model/", default="model/")
    parser.add_argument("-th", "--threshold", type=int, dest="th",
                        help="threshold of word, default is 4", default=4)
    parser.add_argument("-lr", type=float, dest="lr",
                        help="Default is 1e-4", default=1e-4)
    parser.add_argument("-model_type", type=str, dest="mt",
                        help="five types to choose", choices=['dnn', 'multi_label', 'devise', 'heracles'])
    parser.add_argument("-end2end", type=str, dest="end2end", choices=['True', 'False'], default="False")
    return parser

def select_model(args, word2idx):
    use_e2e = True if args.end2end == "True" else False
    if args.mt == 'devise':
        core_model = devise(end2end=True) if use_e2e else devise()
        criterion = nn.MSELoss()
    elif args.mt == 'multi_label':
        core_model = multi_label(len(word2idx), end2end=True) if use_e2e else multi_label(len(word2idx))
        criterion = nn.BCELoss()
    elif args.mt == 'dnn':
        core_model = dnn(len(word2idx), end2end=True) if use_e2e else dnn(len(word2idx))
        criterion = nn.NLLLoss()
    return core_model, criterion

def testing(image_model, core_model, word2idx, word_vec_dict, vec_matrix):
    fix_path = "/home/chihen/NTU/Research/Hashtag/HARRISON/"
    print("You should enter to list element separated by space and follow the format: <usr> <img.jpg> <YYMMDDHHmmss>")
    crop = img_centre_crop()
    input_string = input()
    info = input_string.split()
    
    tmp_path = os.path.join(fix_path, info[1])
    img = Image.open(tmp_path).convert('RGB')
    input_img = crop(img).unsqueeze(0).to(device)

    tmp = info.copy()
    with torch.no_grad():
        img_feat = image_model(input_img)
        pred = core_model(img_feat.squeeze().unsqueeze(0)).detach().cpu()
        top_k_result = top_k(pred, vec_matrix, 10)[0]
        for ele in top_k_result:
            tmp += [idx2word[ele]]
        print(tmp)

if __name__ == "__main__":
    folder_dir = "/home/chihen/NTU/Research/Hashtag/HARRISON/"
    filename = "tag_list.txt"
    args = command().parse_args()

    ### Use K.T. data ###
    # word_vec = np.load('tag_list_word2vec_64.npz')
    # vec_matrix = word_vec['word_embeddings']
    ####################
    
    ### Use C.E. data ###
    word_vec = np.load('data/wordvec.npz')
    vec_matrix = word_vec['wordvec']
    word_vec_dict = {idx:vec for idx,vec in enumerate(vec_matrix)}
    _, word_counter, corpus, word2idx, idx2word = read_tags(folder_dir, filename, args)
    #####################

    print("Total wordvec in dataset: ", vec_matrix.shape, " <num_word, dim>")
    
    ### Use K.T data ###
    # word_counter = word_vec['word_count']
    # word2idx = word_vec['dictionary'].tolist()
    # idx2word = word_vec['reverse_dictionary'].tolist()
    ####################
    if args.mt != 'heracles':
        core_model, loss_fn = select_model(args, word2idx)
        core_model = core_model.to(device)
        print(core_model)
    else:
        heracles = Heracles().to(device)
        labours = Labours().to(device)
        loss_fn = nn.MSELoss()
        print(heracles)
        print(labours)

    if args.mode == "train":
        if args.mt != 'heracles':
            training(args, core_model, folder_dir, word2idx, idx2word, word_vec_dict, vec_matrix, loss_fn)
        else:
            train_heracles(args, heracles, labours, folder_dir, word2idx, idx2word, word_vec_dict, vec_matrix, loss_fn)
    else:
        if args.mt == "devise":
            m_path = os.path.join(args.save, 'best_val')
            core_model.load_state_dict(torch.load(m_path+'_{}.pth'.format(args.mt)))
            core_model.eval()
            while True:
                pass
                # testing(image_model, core_model, word2idx, word_vec_dict, vec_matrix)
