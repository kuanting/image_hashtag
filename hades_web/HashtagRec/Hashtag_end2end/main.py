import numpy as np
import pandas as pd
from scipy.spatial import distance
from PIL import Image, ImageFile
import scipy, argparse, os, sys, csv, io, time

import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset 

from src.img_extractor import img_centre_crop, img_extractor
from src.data_reader import data_reader
from src.img2vec import img2vec
from word2vec import read_tags
from utils.evalution import top_k, f1_score, cos_cdist

from pymongo import MongoClient
from bson.objectid import ObjectId
import time

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def command():
    parser = argparse.ArgumentParser(description='Choose the mode you want')
    parser.add_argument("-m", "--mode", type=str, dest="mode", help="train or test, Default is train", default="train")
    parser.add_argument("-id", "--img_id", type=str, dest="imgID", help="image id, Default is None", default=None)
    parser.add_argument("-bs", "--batch_size", type=int, dest="bs", help="Batch_size of network, Default is 64", default=64)
    parser.add_argument("-epos", "--epochs", type=int, dest="epos", help="Numbers of epochs, Default is 200", default=200)
    parser.add_argument("-s", "--save", type=str, dest="save", help="Path of training model, default is ./HashtagRec/Hashtag_end2end/model/", default="./HashtagRec/Hashtag_end2end/model/")
    parser.add_argument("-th", "--threshold", type=int, dest="th", help="threshold of word, default is 4", default=4)
    parser.add_argument("-lr", type=float, dest="lr", help="Default is 1e-4", default=1e-4)
    return parser

def training(args, img_extractor, img2vec, folder_dir, word2idx, word_vec_dic, vec_matrix):
    tr_img = data_reader(folder_dir, word2idx, word_vec_dict)
    te_img = data_reader(folder_dir, word2idx, word_vec_dict, train=False)
    tr = DataLoader(tr_img, batch_size=args.bs, shuffle=True, num_workers=8)
    te = DataLoader(te_img, batch_size=args.bs, shuffle=True, num_workers=8)

    optimizer_ex = optim.Adam(img_extractor.parameters(), lr=args.lr)
    optimizer_img = optim.Adam(img2vec.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val, best_tr = 0, 0
    for epoch in range(args.epos):
        print('This is epoch {:4d}'.format(epoch))
        total_loss, all_tp, all_total = 0, 0, 0
        img_extractor.train()
        img2vec.train()
        st = time.time()
        for i,(x,y) in enumerate(tr, 1):
            cate_y, hashtag_y = y
            x = x.to(device)
            cate_y = cate_y.to(device)
            
            img_extractor.zero_grad()
            img2vec.zero_grad()

            img_feat = img_extractor(x)
            y_pred = img2vec(img_feat.squeeze())
            loss = criterion(y_pred, cate_y)

            loss.backward()
            optimizer_ex.step()
            optimizer_img.step()
            total_loss += loss.item()
            
            ### find top@k of y_pred and evalution ###
            y_pred_find = y_pred.detach().cpu()
            top_k_result = top_k(y_pred_find, vec_matrix, 10)
            tp, total_l = f1_score(top_k_result, hashtag_y, idx2word)
            all_tp += tp
            all_total += total_l
            recall_rate = all_tp/all_total *100 
            print("Batch: {:4d}/{:4d} loss: {:06.4f}, recall: {:04.2f}%".format(i, len(tr), total_loss/i, recall_rate),
                  end=' '*5+'\r' if i != len(tr) else '\n')
        print('total time: {:>5.2f}s'.format(time.time()-st))
        if recall_rate > best_tr:
            best_tr = recall_rate
            print('saving best training recall model...')
            tr_path = os.path.join(args.save, 'best_tr')
            torch.save(img_extractor.state_dict(), tr_path+'_img_extractor.pth')
            torch.save(img2vec.state_dict(), tr_path+'_img2vec.pth')

        with torch.no_grad():
            vall_tp, vall_total = 0, 0
            img_extractor.eval()
            img2vec.eval()
            for i,(vx,vy) in enumerate(te, 1):
                _, val_hashtag_y = vy
                vx = vx.to(device)

                vimg_feat = img_extractor(vx)
                vy_pred = img2vec(vimg_feat.squeeze()).detach().cpu()
                vtop_k_result = top_k(vy_pred, vec_matrix, 10)
                vtp, vtotal_l = f1_score(vtop_k_result, val_hashtag_y, idx2word)
                vall_tp += vtp
                vall_total += vtotal_l
                vrecall_rate = vall_tp/vall_total *100
                print("Testing set recall rate: {:04.2f}% vall_tp: {} vall_total: {}".format(vrecall_rate, vall_tp, vall_total)
                      , end=' '*5+'\r' if i != len(te) else '\n')
            if vrecall_rate > best_val:
                best_val = vrecall_rate
                print('saving best validation recall model...')
                val_path = os.path.join(args.save, 'best_val')
                torch.save(img_extractor.state_dict(), val_path+'_img_extractor.pth')
                torch.save(img2vec.state_dict(), val_path+'_img2vec.pth')

def testing(imageID, img_extractor, img2vec, word2idx, word_vec_dict, vec_matrix):
    fix_path = "./public/files/"
    crop = img_centre_crop()
    #print("You should enter to list element separated by space and follow the format: <usr> <img.jpg> <YYMMDDHHmmss>")
    #input_string = input()
    #info = input_string.split('_')
    
    #tmp_path = os.path.join(fix_path, info[1])
    tmp_path = os.path.join(fix_path, imageID)
    img = Image.open(tmp_path).convert('RGB')
    input_img = crop(img).unsqueeze(0).to(device)

    #tmp = info.copy()
    tmp = []
    tmp += [imageID]
    with torch.no_grad():
        img_feat = img_extractor(input_img)
        pred = img2vec(img_feat.squeeze().unsqueeze(0)).detach().cpu()
        top_k_result = top_k(pred, vec_matrix, 10)[0]
        for ele in top_k_result:
            tmp += [idx2word[ele]]
        print(tmp)
    return tmp


if __name__ == "__main__":
    start_time = time.time()
    folder_dir = "/data/Hashtag/HARRISON/"
    filename = "tag_list.txt"
    args = command().parse_args()

    img_extractor = img_extractor().to(device)
    img2vec = img2vec().to(device)

    vec_matrix = np.load('./HashtagRec/Hashtag_end2end/wordvec.npz')['wordvec']
    print("Total wordvec in dataset: ", vec_matrix.shape, " <num_word, dim>")
    after_wordvec = time.time()
    word_vec_dict = {idx:vec for idx,vec in enumerate(vec_matrix)}
    _, _, corpus, word2idx, idx2word = read_tags(folder_dir, filename, args)

    if args.mode == "train":
        img_extractor.train()
        img2vec.train()
        training(args, img_extractor, img2vec, folder_dir, word2idx, word_vec_dict, vec_matrix)
    else:
        m_path = os.path.join(args.save, 'best_val')
        img_extractor.load_state_dict(torch.load(m_path+'_img_extractor.pth'))
        img2vec.load_state_dict(torch.load(m_path+'_img2vec.pth'))

        img_extractor.eval()
        img2vec.eval()
	
        result = testing(args.imgID, img_extractor, img2vec, word2idx, word_vec_dict, vec_matrix)
        after_test = time.time()

        conn = MongoClient('127.0.0.1', 27017)
        db = conn.taglist
        collection = db.imageTag
        collection.stats

        # insert tags to mongodb
        collection.insert_one({
                "imgID": result[0],
                "one": result[1],
                "two": result[2],
                "three": result[3],
                "four": result[4],
                "five": result[5],
                "six": result[6],
                "seven": result[7],
                "eight": result[8],
                "nine": result[9],
                "ten": result[10]
            })
        after_db = time.time()
        print("loading wordvec: ", after_wordvec-start_time)
        print("run testing model: ", after_test-after_wordvec)
        print("write to db: ", after_db-after_test)
        print("all: ", after_db-start_time)
