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

class img2vec(nn.Module):
    def __init__(self):
        super(img2vec, self).__init__()
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 500) 
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    model = img2vec()
    print("this is img2vec model structure")
    print(model)