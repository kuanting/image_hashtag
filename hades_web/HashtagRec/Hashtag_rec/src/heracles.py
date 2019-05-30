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

class Heracles(nn.Module):
    def __init__(self):
        super(Heracles, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (1) x 64 x 64
        )
        self.translate = nn.Sequential(
            nn.Linear(25600, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, 500)
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.translate(output)
        return output

class Labours(nn.Module):
    def __init__(self):
        super(Labours, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, 64, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64 * 2, 4, 1, bias=False),
            nn.BatchNorm1d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64 * 2, 64 * 4, 4, 1, bias=False),
            nn.BatchNorm1d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64 * 4, 64 * 8, 4, 1, bias=False),
            nn.BatchNorm1d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64 * 8, 1, 4, 1, bias=False),
        )
        self.distinguish = nn.Sequential(
            nn.Linear(485, 1),
            nn.Sigmoid()
        )
        self.scoring = nn.Sequential(
            nn.Linear(485, 200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        # output = output.squeeze(1)
        output = output.view(output.size(0), -1)
        real_fake = self.distinguish(output)
        score = self.scoring(output)

        return real_fake, score
