import os
import sys
sys.path.append('../../../')
import pandas as pd
from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net
import pickle
from tqdm import tqdm
from bunny import bunny


test_path = os.path.join("/home/liaop20/data/salt", 'test')
#test_path = os.path.join("/data/kaggle/salt", 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
print(test_file_list[:3])

p1 = pickle.load(open("100models_weighted_sum_all_sigmoids_1.p","rb"))

p2 = pickle.load(open("../pseudo.p","rb"))

p=(p1+p2)/2


threshold = 0.45
binary_prediction = (p > threshold).astype(int)


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

all_masks = []
for p_mask in list(binary_prediction):
    p_mask = rle_encoding(p_mask)
    all_masks.append(' '.join(map(str, p_mask)))



submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('./resnet34_pesudo.csv', index = False)
