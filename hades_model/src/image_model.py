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

### Use resnet50 as pretrained model ###
class img_model(nn.Module):
    def __init__(self):
        super(img_model, self).__init__()
        self.conv_part = nn.Sequential(nn.Conv1d(1, 4, 3, stride=2),
                                       nn.BatchNorm1d(4),
                                       nn.Conv1d(4, 8, 3, stride=2),
                                       nn.BatchNorm1d(8),
                                       nn.Conv1d(8, 16, 3, stride=2),
                                       nn.BatchNorm1d(16),
                                       nn.Conv1d(16, 32, 3, stride=2),
                                       nn.BatchNorm1d(32))

        # self.conv_part = nn.Sequential(nn.Conv1d(1, 4, 3, stride=2),
                                       # nn.BatchNorm1d(4),
                                       # nn.ReLU(),
                                       # nn.Conv1d(4, 8, 3, stride=2),
                                       # nn.BatchNorm1d(8),
                                       # nn.ReLU(),
                                       # nn.Conv1d(8, 16, 3, stride=2),
                                       # nn.BatchNorm1d(16),
                                       # nn.Conv1d(16, 32, 3, stride=2),
                                       # nn.BatchNorm1d(32),
                                       # nn.ReLU())

        self.nn_part = nn.Sequential(nn.Linear(127*32, 3072),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(3072),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(3072, 2048))

    def forward(self, x):
        x = self.conv_part(x)
        x = x.reshape(x.size(0), -1)
        x = self.nn_part(x)
        return x

### load the image transformer ###
def img_centre_crop():
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return centre_crop

if __name__ == "__main__":
    model = img_model()
    print("this is img_model structure, using resnet50 as backbone model")
    print(model)
