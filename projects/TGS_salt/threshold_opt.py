import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2

#======================= parameters =======================
# calculate KL with or without zero mask from 0.871
dist_with_zero=False
use_train_annotation=True
thres_start=0.1
thres_end=0.9
thres_step=0.01
bin_num=50
polyfit_order=3
#test sigmoid output here
test_sigmoid_output_file="F:/WorkStation2/tgs_pytorch/output/seresnext50_bn/non_empty_test_10fold/test/final_weighted_merge.prob.uint8.npy"
# test zero out file here
test_zero_out_file="F:/WorkStation2/tgs_pytorch/8519771/10folds_ne_majvote_vert_corrected.csv"
#train or valid sigmoid output here
train_sigmoid_output_file="F:/WorkStation2/tgs_pytorch/output/seresnext50_bn/non_empty_test_10fold/test/merged_submit-1-test_18000-flip.prob.uint8.npy"
#train or valid zero out file here
train_zero_out_file="F:/WorkStation2/tgs_pytorch/8519771/10folds_ne_majvote_vert_corrected.csv"
train_mask_dir="F:/WorkStation2/tgs_pytorch/data/train/masks/"
#======================= parameters =======================




threshold=np.arange(thres_start,thres_end,thres_step)
threshes=[]
KLs=[]
rKLs=[]
zero_out_index=bin_num+1 if dist_with_zero else 0
def get_KL():

    test_edf = pd.read_csv(test_zero_out_file, index_col='id')
    test_sigmoid_output = np.load(test_sigmoid_output_file).astype(np.float32)
    if np.max(test_sigmoid_output)>120:#[0,255] else [0,1]
        test_sigmoid_output=test_sigmoid_output/255.0
    #print(test_sigmoid_output)
    test_zero_index = []
    for n, id in enumerate(test_edf.index):
        if type(test_edf.loc[id]['rle_mask']) == float:
            test_zero_index.append(n)
    #print(test_zero_index)
    print('zero file include '+str(len(test_zero_index))+' zero masks')
    test_zero_index = set(test_zero_index)
    train_edf = pd.read_csv(train_zero_out_file, index_col='id')
    train_sigmoid_output = np.load(train_sigmoid_output_file).astype(np.float32)
    if np.max(train_sigmoid_output)>120:#[0,255] else [0,1]
        train_sigmoid_output=train_sigmoid_output/255.0
    #print(train_sigmoid_output)
    train_zero_index = []
    for n, id in enumerate(train_edf.index):
        if type(train_edf.loc[id]['rle_mask']) == float:
            train_zero_index.append(n)
    #print(train_zero_index)
    print('zero file include ' + str(len(train_zero_index)) + ' zero masks')
    train_zero_index = set(train_zero_index)

    train_mask_list = os.listdir(train_mask_dir)
    train_masks = []
    for l in train_mask_list:
        train_masks.append(cv2.cvtColor(cv2.imread(train_mask_dir + l), cv2.COLOR_BGR2GRAY) / 255)
    train_masks = np.array(train_masks)

    train_dist = _get_cover_distribute(train_masks,set())
    print(train_dist)
    for t in tqdm(threshold):
        test_output=test_sigmoid_output>t
        test_dist=_get_cover_distribute(test_output,test_zero_index)
        if not use_train_annotation:
            train_output=train_sigmoid_output>t
            train_dist=_get_cover_distribute(train_output, train_zero_index)
        # print(test_dist)
        # print(train_dist)
        KL = 0.0
        for i in range(bin_num+1):
            KL += test_dist[i] * np.log(test_dist[i] / train_dist[i])
        reverse_KL = 0.0
        for i in range(bin_num+1):
            reverse_KL += train_dist[i] * np.log(train_dist[i] / test_dist[i])
        threshes.append(t)
        KLs.append(KL)
        rKLs.append(reverse_KL)

def _get_cover_distribute(output,zero_index):
    result=[0]*(bin_num+2)
    for n in range(len(output)):
        if n in zero_index:
            result[zero_out_index]+=1
            continue
        # print(output)
        result[get_coverness(output[n])]+=1
    return np.array(result[:bin_num+1])*1.0/len(output)

def get_coverness(output):
    # print(output.shape)
    # print(np.sum(output[output==1]))
    coverness=np.ceil(bin_num*(np.sum(output[output==1])/(output.shape[0]*output.shape[1])))
    #print(coverness)
    return int(coverness)

get_KL()
import matplotlib.pyplot as plt
plt.plot(threshes,KLs,'*',color='red')
params=np.polyfit(threshes,KLs,polyfit_order)
fitted_threshes=np.arange(thres_start,thres_end,1e-6)
fitted_KLs=np.polyval(params,fitted_threshes)
plt.plot(fitted_threshes,fitted_KLs,'-',color='red')
plt.plot(threshes,rKLs,'*',color='green')
params=np.polyfit(threshes,rKLs,polyfit_order)
fitted_threshes=np.arange(thres_start,thres_end,1e-6)
fitted_rKLs=np.polyval(params,fitted_threshes)
plt.plot(fitted_threshes,fitted_rKLs,'-',color='green')
plt.show()
