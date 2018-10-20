import os
import sys
sys.path.append('../../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net
import pickle
from tqdm import tqdm

train_path = os.path.join('resnet34')
train_file_list = glob.glob(os.path.join(train_path, '*.p'))
train_file_list = [f.split('/')[-1].split('.')[0] for f in train_file_list]

train_path1 = os.path.join('ocnet')
train_file_list1 = glob.glob(os.path.join(train_path1, '*.p'))
train_file_list1 = [f.split('/')[-1].split('.p')[0] for f in train_file_list1]

preds=None
for file1 in tqdm(train_file_list):
   sigmoids=pickle.load(open('resnet34/'+file1+".p","rb"))
   if preds is None: preds=sigmoids
   else: preds=preds+sigmoids

preds=preds/(len(train_file_list))
print(preds.mean())
pickle.dump(preds,open("resnet34_all_sigmoids.p","wb"))



preds2=None

preds2=None
for file1 in tqdm(train_file_list1):
   sigmoids=pickle.load(open('ocnet/'+file1+".p","rb"))
   if preds2 is None: preds2=sigmoids
   else: preds2=preds2+sigmoids
#print(preds2.mean())
preds2=preds2/(len(train_file_list1))
print(preds2.mean())
pickle.dump(preds2,open("ocnet_all_sigmoids.p","wb"))


preds = (preds+preds2)/2
print(preds.mean())
pickle.dump(preds,open("all_sigmoids.p","wb"))
