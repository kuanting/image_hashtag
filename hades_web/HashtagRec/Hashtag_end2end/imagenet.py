import argparse, os, time, sys, csv, io, time
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

def feature_extractor():
    resnet50 = models.resnet50(num_classes=1000, pretrained=True)
    modules = list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)
    return resnet50

def extractor_train(data, tag, folder_dir):
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    resnet50 = feature_extractor().to(device)

    image_feature, total_label, total_hashtag = [], [], []
    st = time.time()
    for idx, (img, label) in enumerate(zip(data, tag), 1):
        print('{:>5d}/{} {}'.format(idx, len(data), img[0]), end=' '*10+'\r' if idx != len(data) else '\n')
        all_hashtag = label[0].strip()
        label = img[0].split('/')[1]
        tmp_name = os.path.join(folder_dir, img[0])
        img = Image.open(tmp_name).convert('RGB')
        input_img = centre_crop(img).unsqueeze(0)
        ### gray image ###
        if input_img.shape[1] == 1:
            print('hh')
            input_img = input_img.detach().numpy()
            fake_img = np.zeros((1, 3, 224, 224))
            for i in range(3):
                fake_img[0, i, :] = input_img[0, 0, :]
            input_img = torch.from_numpy(fake_img).type(torch.FloatTensor)
        input_img = input_img.to(device)
        logit = resnet50.forward(input_img).squeeze().detach().cpu().numpy()

        image_feature.append(logit)
        total_label.append(label)
        total_hashtag.append(all_hashtag)

    np.savez('image_feature.npz', 
             image_feature=image_feature[:-5730], 
             total_label=total_label[:-5730],
             total_hashtag=total_hashtag[:-5730],
             val_image_feature=image_feature[-5730:], 
             val_total_label=total_label[-5730:],
             val_total_hashtag=total_hashtag[-5730:])
    print('total time: ', time.time()-st)

if __name__ == "__main__":
    ### Path of data ###
    folder_dir = "../Hashtag/HARRISON/"
    filename = "tag_list.txt"
    data = pd.read_csv(folder_dir+'data_list.txt', header=None).values
    tag = pd.read_csv(folder_dir+'tag_list.txt', header=None).values
    extractor_train(data, tag, folder_dir)
