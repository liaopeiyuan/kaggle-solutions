from attentionUnet import UnetAttention as Net

directory = '/home/liaop20/salt'

import glob
import lovasz_losses as L


# In[3]:
import cv2

def load_image(path, mask = False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #height, width, _ = img.shape
    height, width = img.shape

    # Padding in needed for UNet models because they need image size to be divisible by 32
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

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img / 255.0
        return torch.from_numpy(img).float()
    else:
        img = img / 255.0
        return torch.from_numpy(img.reshape((1,128,128))).float()

import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split
from tqdm import tqdm,tqdm_notebook
from skimage.transform import resize
from keras.preprocessing.image import load_img
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision



import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils import data

class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test = False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_image(image_path)

        if self.is_test:
            return (image,)
        else:
            mask = load_image(mask_path, mask = True)
            return image, mask


device = "cpu"
model = Net()

initial_checkpoint = '/home/liaop20/Kaggle-TGS/kaggle_tgs/20180826/code/checkpoint/00007000_model.pth'

if initial_checkpoint is not None:
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


test_path = os.path.join(directory, 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
test_file_list[:3], test_path


# In[ ]:


print(len(test_file_list))
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test = True)
model.eval()
all_predictions = []
for image in tqdm(data.DataLoader(test_dataset, batch_size = 75)):
    image = image[0].type(torch.float)
    y_pred = F.sigmoid(model(image)).detach().numpy()
    all_predictions.append(y_pred)
all_predictions_stacked = np.vstack(all_predictions)[:, 0, :, :]


# In[ ]:


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


# In[ ]:


all_predictions_stacked = all_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]


# In[ ]:


all_predictions_stacked.shape


# In[ ]:

depths_df = pd.read_csv(os.path.join(directory, 'train.csv'))

train_path = os.path.join(directory, 'train')
file_list = list(depths_df['id'].values)
print(len(file_list))


# In[10]:


device = "cuda"


# In[11]:


import tqdm

#file_list_val = file_list[::10]
#file_list_train = [f for f in file_list if f not in file_list_val]
#print(type(list(train_df["coverage"].values)))
#print(type(file_list))
#print(train_df["z"].values.shape)

file_list_train, file_list_val= train_test_split(
    file_list,
    test_size=0.1,  random_state=1200)
dataset_val = TGSSaltDataset(train_path, file_list_val)

val_predictions = []
val_masks = []
for image, mask in tqdm.tqdm(data.DataLoader(dataset_val, batch_size = 30)):
    image = image.type(torch.float).to(device)
    y_pred = F.sigmoid(model(image)).cpu().detach().numpy()
    val_predictions.append(y_pred)
    val_masks.append(mask)

val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]

val_masks_stacked = np.vstack(val_masks)[:, :, :]
val_predictions_stacked = val_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]

val_masks_stacked = val_masks_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
val_masks_stacked.shape, val_predictions_stacked.shape


# In[ ]:


from sklearn.metrics import jaccard_similarity_score

metric_by_threshold = []
for threshold in np.linspace(0, 1, 11):
    val_binary_prediction = (val_predictions_stacked > threshold).astype(int)

    iou_values = []
    for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
        iou = jaccard_similarity_score(y_mask.flatten(), p_mask.flatten())
        iou_values.append(iou)
    iou_values = np.array(iou_values)

    accuracies = [
        np.mean(iou_values > iou_threshold)
        for iou_threshold in np.linspace(0.5, 0.95, 10)
    ]
    print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))
    metric_by_threshold.append((np.mean(accuracies), threshold))

best_metric, best_threshold = max(metric_by_threshold)


# In[ ]:


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


# In[ ]:


submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('submit_baseline2.csv.gz', compression = 'gzip', index = False)
