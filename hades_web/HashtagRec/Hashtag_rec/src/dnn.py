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

class dnn(nn.Module):
    def __init__(self, word_len, end2end=False):
        super(dnn, self).__init__()
        self.end2end = end2end
        if end2end:
            import torchvision.models as models
            resnet50 = models.resnet50(num_classes=1000, pretrained=True)
            modules = list(resnet50.children())[:-1]
            self.resnet50 = nn.Sequential(*modules)
        self.model = nn.Sequential(nn.Linear(2048, 2000),
                                   nn.BatchNorm1d(2000),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(2000, 1800),
                                   nn.BatchNorm1d(1800),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1800, 1600),
                                   nn.BatchNorm1d(1600),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1600, 1400),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.BatchNorm1d(1400),
                                   nn.Linear(1400, 1200),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.BatchNorm1d(1200),
                                   nn.Linear(1200, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.BatchNorm1d(1024),
                                   nn.Linear(1024, word_len),
                                   nn.LogSoftmax(dim=-1))
    
    def forward(self, x):
        if self.end2end:
            x = self.resnet50(x)
        x = x.reshape(x.size(0), -1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    model = multi_label(100)
    print("this is nerual network model structure")
    print(model)
