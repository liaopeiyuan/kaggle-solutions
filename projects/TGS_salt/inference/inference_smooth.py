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
EVAL_256 = True
EVAL_128 = True
FOLDS=[]
for i in range(10):
    FOLDS.append(str(i))


RANGE=[]
#RANGE.append([])
#for i in range(3):
#    RANGE.append(np.arange(int(79.5*1000),110*1000,10*1000))
#RANGE.append(np.arange(int(79.5*1000),80*1000,10*1000)) 
RANGE.append(np.arange(int(99.5*1000),130*1000,10*1000)) 
RANGE1=[]
RANGE1.append(np.arange(int(9.5*1000),70*1000,10*1000))
#RANGE1.append(np.arange(int(29.5*1000),30*1000,10*1000))
def valid_augment1(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    image, mask = do_resize2(image, mask, SIZE1, SIZE1)
    image, mask = do_center_pad_to_factor2(image, mask,factor=128)
    return image,mask,index,cache
"""
for i in range(7):
    RANGE.append(np.arange(int(59.5*1000),109*1000,10*1000))
"""
#FOLD=["6","9","0","1","8","7","5","4","3","2"]
#RANGE=[np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(119.5*1000),170*1000,10*1000),np.arange(int(49.5*1000),100*1000,10*1000),np.arange(int(59.5*1000),110*1000,10*1000),np.arange(int(109.5*1000),160*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(89.5*1000),140*1000,10*1000)]
batch_size=30
VAL_FOLD="8"
def valid_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    image, mask = do_resize2(image, mask, SIZE, SIZE)
    image, mask = do_center_pad_to_factor2(image, mask,factor=256)
    return image,mask,index,cache

def load_image(path, factor, mask = False):
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

    #img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img / 255.0
        return torch.from_numpy(img).float()
    else:
        image = img / 255.0
        image, _ = do_resize2(image,image, factor, factor)
        image, _ = do_center_pad_to_factor2(image,image, factor=factor)
        return torch.from_numpy(image.reshape((1,factor,factor))).float()


def validation( net, valid_loader, threshold ):

    valid_num  = 0
    valid_loss = np.zeros(3,np.float32)

    predicts = []
    truths   = []

    for input, truth, index, cache in valid_loader:
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit = data_parallel(net,input) #net(input)
            prob  = F.sigmoid(logit)
            loss  = net.criterion(logit, truth)
            dice  = net.metric(logit, truth)

        batch_size = len(index)
        valid_loss += batch_size*np.array(( loss.item(), dice.item(), 0))
        valid_num += batch_size

        prob  = prob [:,:,Y0:Y1, X0:X1]
        truth = truth[:,:,Y0:Y1, X0:X1]
        prob  = F.avg_pool2d(prob,  kernel_size=2, stride=2)
        truth = F.avg_pool2d(truth, kernel_size=2, stride=2)
        predicts.append(prob.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())

    assert(valid_num == len(valid_loader.sampler))
    valid_loss  = valid_loss/valid_num

    #--------------------------------------------------------
    predicts = np.concatenate(predicts).squeeze()
    truths   = np.concatenate(truths).squeeze()
    precision, result, threshold  = do_kaggle_metric(predicts, truths, threshold)
    valid_loss[2] = precision.mean()

    return valid_loss

"""
valid_dataset1 = TGSDataset('list_test_18000', valid_augment1, 'test')
valid_dataset = TGSDataset('list_test_18000', valid_augment, 'test')
valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = null_collate)

valid_loader1  = DataLoader(
                        valid_dataset1,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = 105,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = null_collate)
"""
from random import randint
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split
from tqdm import tqdm,tqdm_notebook
from skimage.transform import resize
from keras.preprocessing.image import load_img


from torch.utils import data

class InferenceDataset(data.Dataset):
    def __init__(self, root_path, file_list, factor, is_test = False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list
        self.factor = factor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path)
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path)
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_image(image_path, factor=self.factor)
        #image, _ = do_resize2(image,image, SIZE, SIZE)
        #image, _ = do_center_pad_to_factor2(image,image, factor=256)
        if self.is_test:
            return (image,)
        else:
            mask = load_image(mask_path, mask = True,factor=self.factor)
            return image, mask


device = "cuda"
model = Net()

#initial_checkpoint = '/home/liaop20/Kaggle-TGS/kaggle_tgs/20180826/code/checkpoint/fold8/00050000_model_f.pth'
#initial_checkpoint = CHECKPOINTS+'/list_train_3600/0000000_model.pth'
#if initial_checkpoint is not None:
#        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
#        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


test_path = os.path.join('/home/liaop20/ml-arsenal/post-processing/TGS_salt/smoothing')
test_file_list = glob.glob(os.path.join(test_path, '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
test_file_list[:3], test_path
test_file_list.sort(key=float)
print(test_file_list)

train_path = os.path.join(DATA, 'train')
train_file_list = glob.glob(os.path.join(train_path, 'images', '*.png'))
train_file_list = [f.split('/')[-1].split('.')[0] for f in train_file_list]
train_file_list[:3], train_path
 

model=model.cuda()
print(len(test_file_list))
test_dataset = InferenceDataset(test_path, test_file_list, is_test = True, factor = 256)
test_dataset1 = InferenceDataset(test_path, test_file_list, is_test = True, factor = 128)
#train_dataset = InferenceDataset(train_path, train_file_list, is_test = True)
pickle.dump( test_dataset, open( "test_dataset.p", "wb" ) )
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

all_predictions_stacked=[]

i=int(167.5*1000)
if i > 59.5*1000:
    stage="256"
else:
    stage=""


for step in RANGE[0]:
 for fold in FOLDS:
  print("Step: {}, Fold: {}".format(step,fold))
  initial_checkpoint = '/home/liaop20/data/salt/checkpoints/list_train'+fold+'_3600_ne_balanced/ResNet34_res_256'+str(step).zfill(8)+'_model.pth'
  model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
  model.set_mode('test')
  model.eval()
  all_predictions = []

  #EVAL = True
  if EVAL_256:
   for image in tqdm(data.DataLoader(test_dataset, batch_size = batch_size)):
    #print(image) 
    #print(image[0].size())
    image = image[0].type(torch.float).to(device)
    #y_pred1 = cv2.resize(F.sigmoid(model(image).cpu()).detach().numpy(),(batch_size,1,128,128))
    y_pred1 = F.sigmoid(model(image).cpu()).detach().numpy()
    #y_pred1 = F.avg_pool2d(y_pred1,  kernel_size=2, stride=2)
    image = np.flip(image.cpu().numpy(),3).copy()
    image = torch.from_numpy(image).type(torch.float).to(device)
    #image = flip(image,3)
    y_pred2 = F.sigmoid(model(image).cpu()).detach().numpy()
    #y_pred2 = cv2.resize(F.sigmoid(model(image).cpu()).detach().numpy(),(batch_size,1,128,128))
    y_pred2 = np.flip(y_pred2,3)
    #y_pred2 = F.avg_pool2d(y_pred2,  kernel_size=2, stride=2)
    y_pred = (y_pred1+y_pred2)/2
    #print(y_pred.shape)
    all_predictions.append(y_pred)
   if all_predictions_stacked == []:
     all_predictions_stacked=np.vstack(all_predictions)[:, 0, :, :]
     #preds_gmean = np.vstack(all_predictions)[:, 0, :, :]

     images=[]
     for i in tqdm(range(len(all_predictions_stacked))):
       img = cv2.resize(all_predictions_stacked[i,:,:],(128,128))
       images.append(img)

     images =np.array(images)
     pickle.dump( images, open( fold+'_smooth_ResNet34_res_256'+str(step).zfill(8)+'_model.p',"wb"))
     all_predictions_stacked=images
     preds_gmean = images
   else:
     temp=np.vstack(all_predictions)[:, 0, :, :]
     
     images=[]
     for i in tqdm(range(len(all_predictions_stacked))):
       img = cv2.resize(temp[i,:,:],(128,128))
       images.append(img)
     images =np.array(images)
     pickle.dump( images, open( fold+'_smooth_ResNet34_res_256'+str(step).zfill(8)+'_model.p',"wb"))

     all_predictions_stacked= all_predictions_stacked+images
     preds_gmean = preds_gmean*images
  else:
   if all_predictions_stacked == []:
     images=pickle.load( open( fold+'_smooth_ResNet34_res_256'+str(step).zfill(8)+'_model.p',"rb"))
     all_predictions_stacked=images
     preds_gmean = images
   else:
     images=pickle.load( open( fold+'_smooth_ResNet34_res_256'+str(step).zfill(8)+'_model.p',"rb"))
     all_predictions_stacked= all_predictions_stacked+images
     preds_gmean = preds_gmean*images
     print(all_predictions_stacked.shape)
     print(all_predictions_stacked.mean()) 
if EVAL:
 pickle.dump( all_predictions_stacked[0:3000,:,:], open( "256_smooth_sigmoids1.p", "wb" ) )
 pickle.dump( all_predictions_stacked[3000:len(all_predictions_stacked),:,:], open( "256_smooth_sigmoids2.p", "wb" ) )
 pickle.dump(  preds_gmean[0:3000,:,:], open( "256_smooth_preds_gmean_sigmoids3.p", "wb" ) )
 pickle.dump(  preds_gmean[3000:len(all_predictions_stacked),:,:], open( "256_smooth_preds_gmean_sigmoids4.p", "wb" ) )

for step in RANGE1[0]:
 for fold in FOLDS:
  print("Step: {}, Fold: {}".format(step,fold))
  #ResNet34_res_00000000_model.pth
  if fold=="6": place=""
  else: place="_"
  initial_checkpoint = '/home/liaop20/data/salt/checkpoints/list_train'+fold+'_3600_ne_balanced/ResNet34_res'+place+str(step).zfill(8)+'_model.pth'
  model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
  model.set_mode('test')
  model.eval()
  all_predictions = []

  #EVAL = True
  if EVAL_128:
   for image in tqdm(data.DataLoader(test_dataset1, batch_size = 105)):
    #print(image) 
    #print(image[0].size())
    image = image[0].type(torch.float).to(device)
    #y_pred1 = cv2.resize(F.sigmoid(model(image).cpu()).detach().numpy(),(batch_size,1,128,128))
    y_pred1 = F.sigmoid(model(image).cpu()).detach().numpy()
    #y_pred1 = F.avg_pool2d(y_pred1,  kernel_size=2, stride=2)
    image = np.flip(image.cpu().numpy(),3).copy()
    image = torch.from_numpy(image).type(torch.float).to(device)
    #image = flip(image,3)
    y_pred2 = F.sigmoid(model(image).cpu()).detach().numpy()
    #y_pred2 = cv2.resize(F.sigmoid(model(image).cpu()).detach().numpy(),(batch_size,1,128,128))
    y_pred2 = np.flip(y_pred2,3)
    #y_pred2 = F.avg_pool2d(y_pred2,  kernel_size=2, stride=2)
    y_pred = (y_pred1+y_pred2)/2
    #print(y_pred.shape)
    all_predictions.append(y_pred)
   if all_predictions_stacked == []:
     all_predictions_stacked=np.vstack(all_predictions)[:, 0, :, :]
     preds_gmean = np.vstack(all_predictions)[:, 0, :, :]
     pickle.dump( all_predictions_stacked, open( fold+'_smooth_ResNet34_res_'+str(step).zfill(8)+'_model.p',"wb"))
   else:
     temp=np.vstack(all_predictions)[:, 0, :, :]
     all_predictions_stacked= all_predictions_stacked+temp
     preds_gmean = preds_gmean*temp
     pickle.dump( temp, open( fold+'_smooth_ResNet34_res_256'+str(step).zfill(8)+'_model.p',"wb"))

if EVAL:
  pickle.dump( all_predictions_stacked[0:3000,:,:], open( "128_256_smooth_sigmoids1.p", "wb" ) )
  pickle.dump( all_predictions_stacked[3000:len(all_predictions_stacked),:,:], open( "128_256_smooth_sigmoids2.p", "wb" ) )
  pickle.dump(  preds_gmean[0:3000,:,:], open( "128_256_smooth_preds_gmean_sigmoids3.p", "wb" ) )
  pickle.dump(  preds_gmean[3000:len(all_predictions_stacked),:,:], open( "128_256_smooth_preds_gmean_sigmoids4.p", "wb" ) )

else:
 pred1  = pickle.load( open( "128_256_smooth_sigmoids1.p", "rb" ) )
 pred2  = pickle.load( open( "128_256_smooth_sigmoids2.p", "rb" ) )

 all_predictions_stacked= np.vstack((pred1,pred2))

 del pred1,pred2

 pred21  = pickle.load( open( "128_256_smooth_preds_gmean_sigmoids1.p", "rb" ) )
 pred22  = pickle.load( open( "128_256_smooth_preds_gmean_sigmoids2.p", "rb" ) )

 preds_gmean= np.vstack((pred21,pred22))

 del pred21,pred22

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
for i in tqdm(range(len(all_predictions_stacked))):
   img = cv2.resize(all_predictions_stacked[i,:,:],(101,101))
   images.append(img)

all_predictions_stacked =np.array(images)

images=[]
for i in tqdm(range(len(all_predictions_stacked))):
   img = cv2.resize(preds_gmean[i,:,:],(101,101))
   images.append(img)

preds_gmean=np.array(images)
print(all_predictions_stacked.shape)

pickle.dump( all_predictions_stacked, open( 'ResNet34_10folds_smooth_arith_mean.p',"wb"))
pickle.dump( preds_gmean, open( 'ResNet34_10folds_smooth_geo_mean.p',"wb"))
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

