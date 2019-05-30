import argparse, os, time, sys, csv, io, time, random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

### ???
ImageFile.LOAD_TRUNCATED_IMAGES = True

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def feature_extractor(vgg=True):
    if vgg:
        vgg19bn = models.vgg19_bn(pretrained=True)
        modules = list(vgg19bn.children())[:-1]
        vgg19bn = nn.Sequential(*modules)
        return vgg19bn
    else:
        resnet50 = models.resnet50(num_classes=1000, pretrained=True)
        modules = list(resnet50.children())[:-1]
        resnet50 = nn.Sequential(*modules)
        return resnet50

def extractor_train(data, tag, folder_dir, USE_VGG):
    coefficient = 0.33
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    extract = feature_extractor(vgg=USE_VGG).to(device).eval()
    print(extract)

    image_feature, total_label, total_hashtag = [], [], []
    st = time.time()
    with torch.no_grad():
        for idx, (img, label) in enumerate(zip(data, tag), 1):
            print('{:>5d}/{} {}'.format(idx, len(data), img[0]), end=' '*10+'\r' if idx != len(data) else '\n')
            all_hashtag = label[0].strip()
            label = img[0].split('/')[1]
            tmp_name = os.path.join(folder_dir, img[0])
            img = Image.open(tmp_name).convert('RGB')
            input_img = centre_crop(img).unsqueeze(0)

            input_img = input_img.to(device)
            logit = extract.forward(input_img).squeeze().detach().cpu().numpy()

            image_feature.append(logit)
            total_label.append(label)
            total_hashtag.append(all_hashtag)

    rdn_seed = 520
    random.seed(rdn_seed)
    random.shuffle(image_feature)
    random.seed(rdn_seed)
    random.shuffle(total_label)
    random.seed(rdn_seed)
    random.shuffle(total_hashtag)

    bone = 'vgg19bn' if USE_VGG else 'resnet50'

    np.savez('data/image_feature_{}_tr.npz'.format(bone), 
             image_feature=image_feature[:-int(len(data)*coefficient)],
             total_label=total_label[:-int(len(data)*coefficient)],
             total_hashtag=total_hashtag[:-int(len(data)*coefficient)])
    np.savez('data/image_feature_{}_te.npz'.format(bone),
             image_feature=image_feature[-int(len(data)*coefficient):],
             total_label=total_label[-int(len(data)*coefficient):],
             total_hashtag=total_hashtag[-int(len(data)*coefficient):])

    print('total time: ', time.time()-st)

if __name__ == "__main__":
    ### Path of data ###
    command = sys.argv[1]
    USE_VGG = True if command=='True' else False
    folder_dir = "/home/chihen/NTU/Research/Hashtag/HARRISON/"
    filename = "tag_list.txt"
    data = pd.read_csv(folder_dir+'data_list.txt', header=None).values
    tag = pd.read_csv(folder_dir+'tag_list.txt', header=None).values
    extractor_train(data, tag, folder_dir, USE_VGG)
