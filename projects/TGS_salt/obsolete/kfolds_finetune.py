import os
import sys
sys.path.append('../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net
import pickle

all_predictions_stacked = pickle.load(open("test.p","rb"))
height, width = 101, 101

if height % 32 == 0:
    y_min_pad = 0
    y_max_pad = 0
else:
    y_pad = 32 - height % 32
    y_min_pad = int(y_pad / 2)
    y_max_pad = y_pad - y_min_pad

if width % 32 == 0:
    x_min_pad = 0
    x_max_pad = 0
else:
    x_pad = 32 - width % 32
    x_min_pad = int(x_pad / 2)
    x_max_pad = x_pad - x_min_pad


test_path = os.path.join(DATA, 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]




all_predictions_stacked = all_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]





all_predictions_stacked.shape




depths_df = pd.read_csv(os.path.join(DATA, 'train.csv'))

train_path = os.path.join(DATA, 'train')
file_list = list(depths_df['id'].values)
print(len(file_list))


# In[10]:


device = "cuda"


# In[11]:


import tqdm


best_metric=0
best_threshold = 0.4572
threshold = best_threshold
binary_prediction = (all_predictions_stacked > threshold).astype(int)

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
submit.to_csv(RESULT+'/submission.csv.gz', compression = 'gzip', index = False)
