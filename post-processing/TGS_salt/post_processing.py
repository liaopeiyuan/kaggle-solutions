from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import numpy as np
import glob
import os
from PIL import Image
import pickle
import os
from scipy.spatial.distance import hamming
from const import ROOT
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

test_path = os.path.join(ROOT, 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]

train_path = os.path.join(ROOT, 'train')
train_file_list = glob.glob(os.path.join(train_path, 'images', '*.png'))
train_file_list = [f.split('/')[-1].split('.')[0] for f in train_file_list]

