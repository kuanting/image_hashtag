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

from src.image_model import img_model
# from src.data_reader import data_reader
from src.heracles_data_reader import heracles_data_reader
from src.devise import devise
from word2vec import read_tags
from utils.evalution import top_k, f1_score, cos_cdist, log_manager_heracles

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def saving_best_model(args, heracles, labours, record, best_f1, stage):
    mode = 0 if stage=='train' else 1
    if record.recall[mode] > best_f1:
        print('{:06.4f}% -> {:06.4f}%'.format(best_f1, record.recall[mode]))
        print('saving best {} recall model...'.format(stage))
        best_f1 = record.recall[mode]
        save_path = os.path.join(args.save, 'best_f1_{}_{}'.format(stage, args.end2end))
        torch.save(heracles.state_dict(), save_path+'_H_{}.pth'.format(args.mt))
        torch.save(labours.state_dict(), save_path+'_L_{}.pth'.format(args.mt))
    return best_f1

def train_heracles(args, heracles, labours, folder_dir, word2idx, idx2word, word_vec_dict, vec_matrix, loss_fn):
    tr_img = heracles_data_reader(folder_dir, word2idx, word_vec_dict, model_type=args.mt)
    te_img = heracles_data_reader(folder_dir, word2idx, word_vec_dict, model_type=args.mt, train=False)
    tr = DataLoader(tr_img, batch_size=args.bs, shuffle=True, num_workers=8)
    te = DataLoader(te_img, batch_size=args.bs, shuffle=True, num_workers=8)

    ### label values ###
    REAL_LABEL = 1
    FAKE_LABEL = 0
    
    ### Model initialize ###
    heracles.apply(weights_init)
    labours.apply(weights_init)

    optimizer_H = optim.Adam(heracles.parameters(), lr=args.lr)
    optimizer_L = optim.Adam(labours.parameters(), lr=args.lr)
    
    criterion = loss_fn
    print('Model type is heracles, using nn.MSELoss() as criterion')

    best_f1_val, best_f1_tr = 0, 0
    Dx, Dx_s, D_Gz1, D_Gz1_s, D_Gz2, D_Gz2_s = 0, 0, 0, 0, 0, 0
    Rec = log_manager_heracles(args)
    for epoch in range(args.epos):
        print('Model: heracles | Epoch {:4d}'.format(epoch))
        epoch_st = time.time()
        heracles.train()
        labours.train()
        for i, (x, y) in enumerate(tr, 1):

            cate_y, hashtag_y = y
            x = x.to(device)
            cate_y = cate_y.to(device)
            batch_size = cate_y.size(0)
            label = torch.ones((batch_size,)).to(device)
            ### Update labours ###
            for _ in range(1):
                ### Real data loss ###
                r_pred_label, r_score = labours(cate_y.unsqueeze(1))
                Dx = criterion(r_pred_label, label)
                Dx_score = criterion(r_score, label)

                loss_Dx = 1 * Dx + 3 * Dx_score
                loss_Dx.backward()

                ### Fake data loss ###
                label.fill_(FAKE_LABEL)
                generate_vec = heracles(x)
                f_pred_label, f_score = labours(generate_vec.unsqueeze(1))
                D_Gz1 = criterion(f_pred_label, label)
                D_Gz1_score = criterion(f_score, label)

                loss_D_Gz1 = 1 * D_Gz1 + 3 * D_Gz1_score
                loss_D_Gz1.backward()

                optimizer_L.step()
                labours.zero_grad()
                Rec.accumlate_D(Dx.item(), Dx_score.item(), D_Gz1.item(), D_Gz1_score.item())
            for _ in range(5): 
                ### Update heracles ###
                label.fill_(REAL_LABEL)
                generate_vec = heracles(x)
                f_pred_label, f_score = labours(generate_vec.unsqueeze(1))
                D_Gz2 = criterion(f_pred_label, label)
                D_Gz2_score = criterion(f_score, label)
                similarity_loss = criterion(generate_vec, cate_y)

                loss_D_Gz2 = 1 * D_Gz2 + 3 * D_Gz2_score + 5 * similarity_loss
                loss_D_Gz2.backward()

                optimizer_H.step()
                heracles.zero_grad()
                Rec.accumlate_G(D_Gz2.item(), D_Gz2_score.item())

            ### find top@k of heracles generate word vector and evalution ###
            y_pred = generate_vec.detach().cpu()
            top_k_result = top_k(y_pred, vec_matrix, 10)
            tp, total_l = f1_score(top_k_result, hashtag_y, idx2word)
            Rec.accumlate_R(tp, total_l)
            print("Batch: {:4d}/{:4d} D(x): {:06.5f} D(G(z1)): {:06.5f} D(G(z2)): {:06.5f} Recall: {:04.2f}%".format(
                i, len(tr), *Rec.display()), end=' '*5+'\r' if i != len(tr) else '\n')
        print('Total time: {:>5.2f}s'.format(time.time()-epoch_st))
          
        best_f1_tr = saving_best_model(args, heracles, labours, Rec, best_f1_tr, 'train')
        
        with torch.no_grad():
            heracles.eval()
            labours.eval()
            for i,(vx,vy) in enumerate(te, 1):
                val_cate_y, val_hashtag_y = vy
                vx = vx.to(device)
                val_cate_y = val_cate_y.to(device)
                val_batch_size = val_cate_y.size(0)
                
                ### calculate D(x) and D(x) score ###
                vlabel = torch.ones((batch_size,)).to(device) 
                vr_pred_label, vr_score = labours(val_cate_y.unsqueeze(1))
                vDx = criterion(vr_pred_label, vlabel)
                vDx_score = criterion(vr_score, vlabel)

                ### calculate D(z2) and D(z2) score ###
                vgenerate_vec = heracles(vx)
                vf_pred_label, vf_score = labours(vgenerate_vec.unsqueeze(1))
                vD_Gz2 = criterion(vf_pred_label, vlabel)
                vD_Gz2_score = criterion(vf_score, vlabel)

                ### calculate D(z1) and D(z1) score ###
                vlabel.fill_(FAKE_LABEL) 
                vD_Gz1 = criterion(vf_pred_label, vlabel)
                vD_Gz1_score = criterion(vf_score, vlabel)
                
                Rec.accumlate_G(vD_Gz2.item(), vD_Gz2_score.item(), train=False)
                Rec.accumlate_D(vDx.item(), vDx_score.item(), vD_Gz1.item(), vD_Gz1_score.item(), train=False)

                vy_pred = vgenerate_vec.detach().cpu()
                vtop_k_result = top_k(vy_pred, vec_matrix, 10) if args.mt != 'multi_label' else vy_pred
                vtp, vtotal_l = f1_score(vtop_k_result, val_hashtag_y, idx2word)
                
                Rec.accumlate_R(vtp, vtotal_l, train=False)
                print("Test D(x): {:06.5f} D(G(z1)): {:06.5f} D(G(z2)): {:06.5f} Recall: {:04.2f}%".format(
                        *Rec.display(train=False)), end=' '*5+'\r' if i != len(te) else '\n')

            best_f1_val = saving_best_model(args, heracles, labours, Rec, best_f1_val, 'val')

        Rec.record_log()
        Rec.plot_log()

if __name__ == "__main__":
    folder_dir = "/home/chihen/NTU/Hashtag/HARRISON/"
    filename = "tag_list.txt"
    print('this is train') 
