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
from src.non_end2end_datareader import data_reader
from src.devise import devise
from word2vec import read_tags
from utils.evalution import top_k, f1_score, cos_cdist, log_manager

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def saving_best_model(args, core_model, record, best_f1, stage):
    mode = 0 if stage=='train' else 1
    use_end2end = "E2E" if args.end2end == "True" else "nonE2E"
    if record.recall[mode] > best_f1:
        print('{:06.4f}% -> {:06.4f}%'.format(best_f1, record.recall[mode]))
        print('saving best {} recall model...'.format(stage))
        best_f1 = record.recall[mode]
        save_path = os.path.join(args.save, 'best_f1_{}_{}'.format(stage, use_end2end))
        torch.save(core_model.state_dict(), save_path+'_{}.pth'.format(args.mt))
    return best_f1

def handle_output_format(args, y_pred, word_vec_dict):
    if args.mt == 'multi_label' or args.mt == 'dnn':
        tmp_y = np.argsort(y_pred.detach().cpu().numpy(), axis=1)[:, -10:]
        if args.mt == 'dnn':
            tmp_vec = np.empty((len(y_pred), len(word_vec_dict[0])))
            for tmp_idx in range(len(tmp_y)):
                first_idx = tmp_y[tmp_idx, -1]
                tmp_vec[tmp_idx, :] = word_vec_dict[first_idx]
            tmp_y = tmp_vec
    else:
        tmp_y = y_pred.detach().cpu()

    return tmp_y

def training(args, core_model, folder_dir, word2idx, idx2word, word_vec_dict, vec_matrix, loss_fn):
    if args.end2end == 'True':
        from src.data_reader import data_reader
        print('Use end2end...')
    else:
        from src.non_end2end_datareader import data_reader
        print('Use non-end2end...')
        core_model.apply(weights_init)
    tr_img = data_reader(folder_dir, word2idx, word_vec_dict, model_type=args.mt)
    te_img = data_reader(folder_dir, word2idx, word_vec_dict, model_type=args.mt, train=False)
    tr = DataLoader(tr_img, batch_size=args.bs, shuffle=True, num_workers=8)
    te = DataLoader(te_img, batch_size=args.bs, shuffle=True, num_workers=8)

    optimizer_tag = optim.Adam(core_model.parameters(), lr=args.lr)
    
    criterion = loss_fn
    print('model_type is {}, using {} as criterion'.format(args.mt, criterion))

    best_f1_val, best_f1_tr = 0, 0
    data_recorder = log_manager(args)
    for epoch in range(args.epos):
        print('Model: {} | Epoch {:4d}'.format(args.mt, epoch))
        epoch_st = time.time()
        core_model.train()

        for i, (x, y) in enumerate(tr, 1):
            cate_y, hashtag_y = y
            x = x.to(device)
            cate_y = cate_y.to(device)
            
            core_model.zero_grad()

            y_pred = core_model(x)
            y_pred = y_pred.type(torch.cuda.DoubleTensor) if args.mt == 'multi_label' else y_pred
            loss = criterion(y_pred, cate_y)

            loss.backward()
            optimizer_tag.step()
            
            ### find top@k of y_pred and evalution ###
            y_pred = handle_output_format(args, y_pred, word_vec_dict)
            top_k_result = top_k(y_pred, vec_matrix, 10) if args.mt != 'multi_label' else y_pred
            tp, total_l = f1_score(top_k_result, hashtag_y, idx2word)
            
            data_recorder.accumulate(loss.item(), tp, total_l)
 
            print("Batch: {:4d}/{:4d} loss: {:06.5f}, recall: {:04.2f}%".format(i, len(tr),
                data_recorder.loss[0], data_recorder.recall[0]), end=' '*5+'\r' if i != len(tr) else '\n')
        print('total time: {:>5.2f}s'.format(time.time()-epoch_st))

        best_f1_tr = saving_best_model(args, core_model, data_recorder, best_f1_tr, 'train')

        with torch.no_grad():
            core_model.eval()
            for i,(vx,vy) in enumerate(te, 1):
                val_cate_y, val_hashtag_y = vy
                vx = vx.to(device)
                val_cate_y = val_cate_y.to(device)

                vy_pred = core_model(vx)
                vy_pred = vy_pred.type(torch.cuda.DoubleTensor) if args.mt == 'multi_label' else vy_pred
                vloss = criterion(vy_pred, val_cate_y)

                vy_pred = handle_output_format(args, vy_pred, word_vec_dict)
                vtop_k_result = top_k(vy_pred, vec_matrix, 10) if args.mt != 'multi_label' else vy_pred
                vtp, vtotal_l = f1_score(vtop_k_result, val_hashtag_y, idx2word)
                
                data_recorder.accumulate(vloss.item(), vtp, vtotal_l, train=False)

                print("Testing set val_loss: {:06.5f} val_recall: {:04.2f}%".format(
                    data_recorder.loss[1], data_recorder.recall[1]), end=' '*5+'\r' if i != len(te) else '\n')
            
            best_f1_val = saving_best_model(args, core_model, data_recorder, best_f1_val, 'val')
        data_recorder.record_log()
        data_recorder.plot_log()

if __name__ == "__main__":
    folder_dir = "/home/chihen/NTU/Hashtag/HARRISON/"
    filename = "tag_list.txt"
    print('this is train') 
