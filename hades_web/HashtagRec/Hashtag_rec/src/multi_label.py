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

class multi_label(nn.Module):
    def __init__(self, word_len, end2end=False):
        super(multi_label, self).__init__()
        self.end2end = end2end
        if end2end:
            import torchvision.models as models
            resnet50 = models.resnet50(num_classes=1000, pretrained=True)
            modules = list(resnet50.children())[:-1]
            self.resnet50 = nn.Sequential(*modules)
        self.cpart = nn.Sequential(nn.Conv1d(1, 32, 5, stride=2),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2, stride=2),
                                   nn.Dropout(p=0.2),
                                   nn.Conv1d(32, 64, 5, stride=2),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2, stride=2),
                                   nn.Dropout(p=0.3),
                                   nn.Conv1d(64, 128, 5, stride=2),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2, stride=2),
                                   nn.Dropout(p=0.3),
                                   nn.Conv1d(128, 256, 5, stride=2),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2, stride=2),
                                   nn.Dropout(p=0.3))

        self.nnpart = nn.Sequential(nn.Linear(256*7, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(128, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(128, word_len),
                                    nn.Softmax(dim=-1))
        
    def forward(self, x):
        if self.end2end:
            x = self.resnet50(x)
            x = x.view(x.size(0), 1 ,-1)
        x = self.cpart(x)
        x = x.view((-1, 256*7))
        x = self.nnpart(x)
        return x


if __name__ == "__main__":
    model = multi_label(100)
    print("this is multi_label model structure")
    print(model)
