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

mode = "100models_weighted"
TUNE=True
df = pd.read_csv("stable.csv")
file_list = list(df["Unnamed: 0"])
print(file_list)
preds1=None
preds2=None
preds3=None
preds4=None

file_list_2 = glob.glob(os.path.join('/home/liaop20/ml-arsenal/projects/TGS_salt/ocnet256',"*.p"))
#file_list = [f.split('/')[-1].split('.npy')[0] for f in file_list]

test_path = os.path.join("/home/liaop20/data/salt", 'test')
#test_path = os.path.join("/data/kaggle/salt", 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
print(test_file_list[:3])
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

<<<<<<< HEAD
#R1='/home/liaop20/ml-arsenal/projects/TGS_salt/100models/'
#R2='/home/liaop20/ml-arsenal/projects/TGS_salt/ocnet/'
R1="/data/liao_checkpoints/100models/"
R2="/data/liao_checkpoints/ocnet/"
R3=
=======


height1, width1 = 202, 202

if height1 % 256 == 0:
    y_min_pad1 = 0
    y_max_pad1 = 0
else:
    y_pad1 = 256 - height1 % 256
    y_min_pad1 = int(y_pad1 / 2)
    y_max_pad1 = y_pad1 - y_min_pad1

if width1 % 256 == 0:
    x_min_pad1 = 0
    x_max_pad1 = 0
else:
    x_pad1 = 256 - width1 % 256
    x_min_pad1 = int(x_pad1 / 2)
    x_max_pad1 = x_pad1 - x_min_pad1


R1='/data/liao_checkpoints/100models/'
R2='/data/liao_checkpoints/ocnet/'
R3='/data/liao_checkpoints/ocnet256/'
#R1="/data/liao_checkpoints/100models/"
#R2="/data/liao_checkpoints/ocnet/"
>>>>>>> 8006bb3b316278a44ddec10efcea2a29bb76b9b2
#1_ResNet34_res_25600029500_model.p
#ResNet34_res_256_fold200119500_model.p'
ct1=0
ct2=0
ct3=0
ct4=0
if TUNE:

 """
 for file1 in tqdm(file_list_2):
     sigmoid_4 = np.load(file1)/255
     ct4=ct4+1
     if preds4 is None:
         preds4=sigmoid_4
         #print(preds4.shape)
     else:
         preds4=preds4+sigmoid_4
         print(preds4.mean()/ct4)
 """
 for file1 in tqdm(file_list):
   #sigmoids=pickle.load(open(file1+".p","rb"))
   #print(file1)
   model = ""
   if file1.find("OCnet")==-1 and file1.find("256")!=-1:
       fold = file1[file1.find("fold")+4:file1.find("fold")+5]
       checkpt= file1[file1.find("fold")+5:file1.find("_simu")]
       #size = file1
       sigmoids_1=pickle.load(open(R1+fold+"_ResNet34_res_256"+checkpt+"_model.p","rb"))
       images=[]
       for i in range(18000):
           img = cv2.resize(sigmoids_1[i,:,:],(101,101))
           images.append(img)
       sigmoids_1 = np.array(images)
       ct1=ct1+1
       model = "256"
   elif file1.find("OCnet")== -1:
       fold = file1[file1.find("fold")+4:file1.find("fold")+5]
       checkpt= file1[file1.find("fold")+5:file1.find("_simu")]
       #size = file1
       sigmoids_2=pickle.load(open(R1+fold+"_ResNet34_res_256"+checkpt+"_model.p","rb"))
       images=[]
       for i in range(18000):
           img = cv2.resize(sigmoids_2[i,:,:],(101,101))
           images.append(img)
       sigmoids_2 = np.array(images)
       ct2=ct2+1
       model = "128"
<<<<<<< HEAD
   elif file1.find("OCnet")!= -1:
       fold = file1[file1.find("fold")+4:file1.find("fold")+5]
       checkpt= file1[file1.find("fold")+5:file1.find("_simu")]
       #size = file1
       sigmoids_4=pickle.load(open(R1+fold+"_ResNet34_res_256"+checkpt+"_model.p","rb"))
=======
   elif file1.find("OCnet256")!=-1:
       fold= file1[file1.find("fold")+4:file1.find("fold")+5]
       checkpt= file1[file1.find("fold")+5:file1.find("_simu")]
       sigmoids_4=pickle.load(open(R3+"ocnet256"+fold+checkpt+".p","rb")) 
       sigmoids_4=sigmoids_4[:, y_min_pad1:256 - y_max_pad1, x_min_pad1:256 - x_max_pad1]
       
>>>>>>> 8006bb3b316278a44ddec10efcea2a29bb76b9b2
       images=[]
       for i in range(18000):
           img = cv2.resize(sigmoids_4[i,:,:],(101,101))
           images.append(img)
       sigmoids_4 = np.array(images)
<<<<<<< HEAD
=======

>>>>>>> 8006bb3b316278a44ddec10efcea2a29bb76b9b2
       ct4=ct4+1
       model = "ocnet256"
   else:
       fold= file1[file1.find("fold")+4:file1.find("fold")+5]
       checkpt= file1[file1.find("fold")+5:file1.find("_simu")]
       sigmoids_3=pickle.load(open(R2+"ocnet"+fold+checkpt+".p","rb")) 
       sigmoids_3=sigmoids_3[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]    
       ct3=ct3+1
       model = "ocnet"

   i=0
   if preds1 is None and model == "256": 
     i=i+1
     preds1=sigmoids_1
   elif model == "256": 
     i=i+1
     preds1=preds1+sigmoids_1
     print(preds1.mean()/ct1)
   
   if preds2 is None and model == "128": 
     i=i+1
     preds2=sigmoids_2
   elif model == "128": 
     i=i+1
     preds2=preds2+sigmoids_2
     print(preds2.mean()/ct2)
    
   if preds3 is None and model == "ocnet": 
     i=i+1
     preds3=sigmoids_3
   elif model=="ocnet": 
     i=i+1     
     preds3=preds3+sigmoids_3
     print(preds3.mean()/ct3)


   if preds4 is None and model == "ocnet256":
     i=i+1
     preds4=sigmoids_4
   elif model=="ocnet256":
     i=i+1
     preds4=preds4+sigmoids_4
     print(preds4.mean()/ct4)
   assert(i==1)
 preds1=preds1/ct1
 preds2=preds2/ct2
 preds3=preds3/ct3
 preds4=preds4/ct4
 pickle.dump(preds1,open( mode+"_sum_all_sigmoids_1.p","wb"))
 pickle.dump(preds2,open( mode+"_sum_all_sigmoids_2.p","wb"))
 pickle.dump(preds3,open( mode+"_sum_all_sigmoids_3.p","wb"))
 pickle.dump(preds3,open( mode+"_sum_all_sigmoids_4.p","wb"))
if not(TUNE):
 preds1=pickle.load(open( mode+"_sum_all_sigmoids_1.p","rb"))
 preds2=pickle.load(open( mode+"_sum_all_sigmoids_2.p","rb"))
 preds3=pickle.load(open( mode+"_sum_all_sigmoids_3.p","rb"))
 preds4=pickle.load(open( mode+"_sum_all_sigmoids_4.p","rb"))
"""
 print(tuple2)

 preds1 = tuple1[0]
 ct1= tuple1[1]
 preds2 = tuple2[0]
 ct2= tuple2[1]
 preds3 = tuple3[0]
 ct3= tuple3[1]

print(preds1[0].shape)
print((ct1,ct2,ct3))

print(preds2)
for i in tqdm(range(len(preds1))):
    preds1[i,:,:]/=ct1
    preds2[i,:,:]/=ct2
    preds3[i,:,:]/=ct3

#preds1=np.true_divide(preds1[0],ct1)
#preds2=np.true_divide(preds2,ct2)
#preds3=np.true_divide(preds3,ct3)
"""
print(preds1.mean())
print(preds2.mean())
print(preds3.mean())
print(preds4.mean())

preds = preds1*0.35+preds2*0.1+preds3*0.1+preds4*0.45
print(preds.mean())
pickle.dump(preds,open("stable_all_sigmoids.p","wb"))

threshold = 0.45
binary_prediction = (preds > threshold).astype(int)


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
submit.to_csv('./100_stable_models_weighted.csv', index = False)

