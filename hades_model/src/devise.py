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

class devise(nn.Module):
    def __init__(self, end2end=False):
        super(devise, self).__init__()
        self.end2end = end2end
        if end2end:
            import torchvision.models as models
            resnet50 = models.resnet50(num_classes=1000, pretrained=True)
            modules = list(resnet50.children())[:-1]
            self.resnet50 = nn.Sequential(*modules)
        else:
            self.conv_part = nn.Sequential(nn.Conv1d(1, 4, 3, stride=2),
                                           nn.BatchNorm1d(4),
                                           nn.Conv1d(4, 8, 3, stride=2),
                                           nn.BatchNorm1d(8),
                                           nn.Conv1d(8, 16, 3, stride=2),
                                           nn.BatchNorm1d(16),
                                           nn.Conv1d(16, 32, 3, stride=2),
                                           nn.BatchNorm1d(32))
            self.nn_part = nn.Sequential(nn.Linear(127*32, 3072),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(3072),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(3072, 2048))
        self.model = nn.Sequential(nn.Linear(2048, 1024),
                                   nn.Linear(1024, 500))

    def forward(self, x):
        if self.end2end:
            x = self.resnet50(x)
            x = x.reshape(x.size(0), -1)
            x = self.model(x)
            return x
        else:
            x = self.conv_part(x.view((x.size(0), 1, -1)))
            x = x.reshape(x.size(0), -1)
            x = self.nn_part(x)
            x = self.model(torch.squeeze(x))
            return x


if __name__ == "__main__":
    model = devise()
    print("this is devise model structure")
    print(model)
