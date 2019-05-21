import argparse, os, time, sys, csv, io, time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

### ??? ###
ImageFile.LOAD_TRUNCATED_IMAGES = True

### cuda gpu ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def img_extractor():
    resnet50 = models.resnet50(num_classes=1000, pretrained=True)
    modules = list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)
    return resnet50

def img_centre_crop():
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return centre_crop

if __name__ == "__main__":
    model = img_extractor()
    print("this is img_extractor structure, using resnet50 as backbone model")
    print(model)