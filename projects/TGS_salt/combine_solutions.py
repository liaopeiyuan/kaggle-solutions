import os
import sys
sys.path.append('../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net
import pickle
TUNE = False
SIZE1=128
PAD1=128

SIZE = 256
PAD  = 256
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,
#FOLD=["0","1","2"]
EVAL = True
EVAL_256 = False
EVAL_128 = False
FOLDS=[]
for i in range(10):
    FOLDS.append(str(i))
from tqdm import tqdm

RANGE=[]
#for i in range(3):
#    RANGE.append(np.arange(int(79.5*1000),110*1000,10*1000))
#RANGE.append(np.arange(int(79.5*1000),80*1000,10*1000)) 
RANGE.append(np.arange(int(79.5*1000),130*1000,10*1000))
RANGE1=[]
RANGE1.append(np.arange(int(29.5*1000),70*1000,10*1000))


pred1  = pickle.load( open( "128_256_sigmoids1.p", "rb" ) )
pred2  = pickle.load( open( "128_256_sigmoids2.p", "rb" ) )
pred3  = pickle.load( open( "128_256_sigmoids3.p", "rb" ) )
pred4  = pickle.load( open( "128_256_sigmoids4.p", "rb" ) )
pred5  = pickle.load( open( "128_256_sigmoids5.p", "rb" ) )
pred6  = pickle.load( open( "128_256_sigmoids6.p", "rb" ) )


all_predictions_stacked= np.vstack((pred1,pred2,pred3,pred4,pred5,pred6))
del pred1,pred2,pred3,pred4,pred5,pred6

pred21  = pickle.load( open( "128_256_preds_gmean_sigmoids1.p", "rb" ) )
pred22  = pickle.load( open( "128_256_preds_gmean_sigmoids2.p", "rb" ) )
pred23  = pickle.load( open( "128_256_preds_gmean_sigmoids3.p", "rb" ) )
pred24  = pickle.load( open( "128_256_preds_gmean_sigmoids4.p", "rb" ) )
pred25  = pickle.load( open( "128_256_preds_gmean_sigmoids5.p", "rb" ) )
pred26  = pickle.load( open( "128_256_preds_gmean_sigmoids6.p", "rb" ) )

preds_gmean= np.vstack((pred21,pred22,pred23,pred24,pred25,pred26))

del pred21,pred22,pred23,pred24,pred25,pred26

#sums=0
#for alist in RANGE:
#    sums=sums+len(alist)
sums=len(RANGE[0])*len(FOLDS)+len(RANGE1[0])*len(FOLDS)
all_predictions_stacked=all_predictions_stacked/(sums)
preds_gmean = np.float_power(preds_gmean,float(1)/float(sums))

print(all_predictions_stacked.shape)


#pickle.dump( all_predictions_stacked, open( 'ResNet34_10folds_arith_mean.p',"wb"))
#pickle.dump( preds_gmean, open( 'ResNet34_10folds_geo_mean.p',"wb"))
#all_predictions_stacked = all_predictions_stacked[:, 27:256 - 27, 27:256 - 27]



print(all_predictions_stacked.shape)

images=[]
for i in tqdm(range(18000)):
   img = cv2.resize(all_predictions_stacked[i,:,:],(101,101))
   images.append(img)

all_predictions_stacked =np.array(images)

images=[]
for i in tqdm(range(18000)):
   img = cv2.resize(preds_gmean[i,:,:],(101,101))
   images.append(img)

preds_gmean=np.array(images)
print(all_predictions_stacked.shape)

pickle.dump( all_predictions_stacked, open( 'ResNet34_10folds_arith_mean.p',"wb"))
pickle.dump( preds_gmean, open( 'ResNet34_10folds_geo_mean.p',"wb"))
"""
image = torch.from_numpy(all_predictions_stacked).type(torch.float)
image = F.avg_pool2d(image,  kernel_size=2, stride=2)

all_predictions_stacked = image.cpu().detach().numpy()
"""

depths_df = pd.read_csv(os.path.join(DATA, 'train.csv'))

train_path = os.path.join(DATA, 'train')
file_list = list(depths_df['id'].values)
print(len(file_list))



threshold = 0.45
binary_prediction = (all_predictions_stacked > threshold).astype(int)

binary_prediction2 = (preds_gmean > threshold).astype(int)

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

all_masks2 = []
for p_mask in list(binary_prediction2):
    p_mask = rle_encoding(p_mask)
    all_masks2.append(' '.join(map(str, p_mask)))


submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv(RESULT+'/128_256_arith_mean.csv', index = False)

submit = pd.DataFrame([test_file_list, all_masks2]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv(RESULT+'/128_256_geo_mean.csv', index = False)
