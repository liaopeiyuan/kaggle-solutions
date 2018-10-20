from __future__ import absolute_import
from __future__ import division
from settings import *

print("Importing numerical libraries...")
import numpy as np
import random
import math
import numpy as np
import random
import PIL
import cv2


print("Importing standard libraries...")
if GRAPHICS:
    import matplotlib
    matplotlib.use('TkAgg')
    #matplotlib.use('WXAgg')
    #matplotlib.use('Qt4Agg')
    #matplotlib.use('Qt5Agg') #Qt4Agg
    print("\tmatplotlib backend: {}".format(matplotlib.get_backend()))
# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
from collections import OrderedDict


print("Importing miscellaneous functions...")
class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def read_list_from_file(data):
    with open(data) as file:
        lines = [line.strip() for line in file]
    return lines

print("Importing constants...")
PI  = np.pi
INF = np.inf
EPS = 1e-12



print("Importing Neural Network dependencies...")
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils import model_zoo
from torch.autograd.variable import Variable
print("\tPyTorch")


import keras
print("\tKeras")

import tensorflow
import tensorboard
print("\tTensorFlow")


from training.metric import *
from training.loss import *
from training.lr_scheduler import *
print("\tMetrics, Losses and LR Schedulers")
from training.kaggle_metrics import *
print("\tKaggle Metrics")

from vision.augmentations import *
print("\tImage augmentations")

from datasets.sampler import *
from datasets.TGS_salt.TGSDataset import *
print("\tDatasets")

print("Importing external libraries...")
import external.lovasz_losses as L
print("\tLovasz Losses (elu+1)")
from external.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
print("\tSynchronized BatchNorm2d")
print('')

