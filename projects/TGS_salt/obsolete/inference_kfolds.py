import os
import sys
sys.path.append('../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net
import pickle
TUNE = False
SIZE = 101
PAD  = 27
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,
FOLD=["0","1","2","3","9"]

RANGE=[]
for i in range(5):
    RANGE.append(np.arange(int(59.5*1000),109*1000,10*1000))
#FOLD=["6","9","0","1","8","7","5","4","3","2"]
#RANGE=[np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(119.5*1000),170*1000,10*1000),np.arange(int(49.5*1000),100*1000,10*1000),np.arange(int(59.5*1000),110*1000,10*1000),np.arange(int(109.5*1000),160*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(89.5*1000),140*1000,10*1000)]
batch_size=30
VAL_FOLD="8"
def valid_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    image, mask = do_resize2(image, mask, SIZE, SIZE)
    image, mask = do_center_pad_to_factor2(image, mask)
    return image,mask,index,cache
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



valid_dataset = TGSDataset('list_train'+VAL_FOLD+'_3600', valid_augment, 'train')
valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = 40,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = null_collate)

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


device = "cuda"
model = Net()

#initial_checkpoint = '/home/liaop20/Kaggle-TGS/kaggle_tgs/20180826/code/checkpoint/fold8/00050000_model_f.pth'
#initial_checkpoint = CHECKPOINTS+'/list_train_3600/0000000_model.pth'
#if initial_checkpoint is not None:
#        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
#        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


test_path = os.path.join(DATA, 'test')
test_file_list = glob.glob(os.path.join(test_path, 'images', '*.png'))
test_file_list = [f.split('/')[-1].split('.')[0] for f in test_file_list]
test_file_list[:3], test_path


train_path = os.path.join(DATA, 'train')
train_file_list = glob.glob(os.path.join(train_path, 'images', '*.png'))
train_file_list = [f.split('/')[-1].split('.')[0] for f in train_file_list]
train_file_list[:3], train_path
 

model=model.cuda()
print(len(test_file_list))
test_dataset = InferenceDataset(test_path, test_file_list, is_test = True)
train_dataset = InferenceDataset(train_path, train_file_list, is_test = True)

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

all_predictions_stacked=[]
m=0
for fold in FOLD:
#for fold in ["8"]:
 for i in RANGE[m]:
 #for i in np.arange(200,460,50):
    #i= round(float(i) / 1000) * 1000
    initial_checkpoint = CHECKPOINTS+'/list_train'+fold+'_3600/'+str(i).zfill(8)+'_model.pth'
    model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    model.set_mode('test')
    model.eval()
    all_predictions = []
    for image in tqdm(data.DataLoader(test_dataset, batch_size = 65)):
        #print(image[0].size())
        image = image[0].type(torch.float).to(device)
        y_pred1 = F.sigmoid(model(image).cpu()).detach().numpy()
        image = np.flip(image.cpu().numpy(),3).copy()
        image = torch.from_numpy(image).type(torch.float).to(device)
        #image = flip(image,3)
        y_pred2 = F.sigmoid(model(image).cpu()).detach().numpy()
        y_pred2 = np.flip(y_pred2,3)
        y_pred = (y_pred1+y_pred2)/2
        all_predictions.append(y_pred)
    if all_predictions_stacked == []:
        all_predictions_stacked=np.vstack(all_predictions)[:, 0, :, :]
    else:
        all_predictions_stacked= all_predictions_stacked+np.vstack(all_predictions)[:, 0, :, :]
 m=m+1
#all_predictions_stacked=all_predictions_stacked/len(np.arange(200,460,50))

pickle.dump( all_predictions_stacked, open( "test_temp.p", "wb" ) )


sums=0
for alist in RANGE:
    sums=sums+len(alist)
all_predictions_stacked=all_predictions_stacked/(sums)

pickle.dump( all_predictions_stacked, open( "test.p", "wb" ) )


#all_predictions_stacked=np.mean(all_predictions_stacked,axis=0)
print(all_predictions_stacked.shape)
 
train_predictions_stacked=[]
VALID=False
if VALID:
 m=0
 for fold in ["8"]:
 # for i in RANGE[m][1]:
 #for fold in FOLD:
  for i in RANGE[m]:
     initial_checkpoint = CHECKPOINTS+'/list_train'+fold+'_3600/'+str(i).zfill(8)+'_model.pth'
     model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
     model.set_mode('test')
     model.eval()
     all_predictions = []
     for image in tqdm(data.DataLoader(train_dataset, batch_size = 100)):
         #image = image[0].type(torch.float).to(device)
         #y_pred = F.sigmoid(model(image).cpu()).detach().numpy()
         #all_predictions.append(y_pred)
         image = image[0].type(torch.float).to(device)
         y_pred1 = F.sigmoid(model(image).cpu()).detach().numpy()
         image = flip(image,3)
         #image = np.flip(image.cpu().numpy(),3).copy()
         #image = torch.from_numpy(image).type(torch.float).to(device)
         y_pred2 = F.sigmoid(model(image).cpu()).detach().numpy()
         y_pred2 = np.flip(y_pred2,3)
         y_pred = (y_pred1+y_pred2)/2
         all_predictions.append(y_pred)
     if train_predictions_stacked == []:
         train_predictions_stacked=np.vstack(all_predictions)[:, 0, :, :]
     else:
         train_predictions_stacked= train_predictions_stacked+np.vstack(all_predictions)[:, 0, :, :]
  m=m+1
 train_predictions_stacked=train_predictions_stacked/(len(RANGE)*len(FOLD))
 pickle.dump( train_predictions_stacked, open( "train.p", "wb" ) )
 #train_dataset = InferenceDataset(train_path, train_file_list, is_test = True)

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

#file_list_val = file_list[::10]
#file_list_train = [f for f in file_list if f not in file_list_val]
#print(type(list(train_df["coverage"].values)))
#print(type(file_list))
#print(train_df["z"].values.shape)

"""
file_list_train, file_list_val= train_test_split(
    file_list,
    test_size=0.3)
dataset_val = InferenceDataset(train_path, file_list)

file_list_val = read_list_from_file(CODE+'/datasets/TGS_salt/splits/list_valid'+FOLD+'_400')
new_list=[]
for files in file_list_val:
   #print(files)
   new_list.append(files[files.find('/')+1:len(files)])
   #print(files)
dataset_val = InferenceDataset(train_path, new_list)


val_predictions = []
val_masks = []
val_predictions_stacked=[]

for i in RANGE:
    val_predictions=[]
    val_masks=[]
    initial_checkpoint = CHECKPOINTS+'/list'+FOLD+'_train_3600/'+str(i).zfill(8)+'_model.pth'
    model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    for image, mask in tqdm.tqdm(data.DataLoader(dataset_val, batch_size = 30)):
        image = image.type(torch.float).to(device)
        y_pred = F.sigmoid(model(image)).cpu().detach().numpy()
        val_predictions.append(y_pred)
        val_masks.append(mask)

    val_predictions_stacked.append(np.vstack(val_predictions)[:, 0, :, :])

val_predictions_stacked=np.array(val_predictions_stacked)
val_predictions_stacked=np.mean(val_predictions_stacked,axis=0)

val_masks_stacked = np.vstack(val_masks)[:, :, :]
val_predictions_stacked = val_predictions_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]

val_masks_stacked = val_masks_stacked[:, y_min_pad:128 - y_max_pad, x_min_pad:128 - x_max_pad]
val_masks_stacked.shape, val_predictions_stacked.shape

from sklearn.metrics import jaccard_similarity_score
"""
metric_by_threshold = []
if TUNE:
 for threshold in np.linspace(0.4, 0.6, 50):
    """
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
    prec_vals=[]
    for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
        predicts = np.concatenate(p_mask).squeeze()
        truths   = np.concatenate(y_mask).squeeze()
        precision, result, t  = do_kaggle_metric(predicts, truths, threshold=threshold)
        prec_vals.append(precision)
    #precision, result, threshold   = do_kaggle_metric(p_mask,y_mask)
    res=np.array(prec_vals).mean()
    #print(res)"""
    predicts=[]
    truths=[]
    for i in RANGE:
        val_predictions=[]
        val_masks=[]
        initial_checkpoint = CHECKPOINTS+'/list_train'+FOLD+'_3600/'+str(i).zfill(8)+'_model.pth'
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        
        valid_num  = 0
        valid_loss = np.zeros(3,np.float32)

        predicts_l = []
        truths_l   = []
        
        for input, truth, index, cache in valid_loader:
            input = input.cuda()
            truth = truth.cuda()
            with torch.no_grad():
                logit = data_parallel(model,input) #net(input)
                prob  = F.sigmoid(logit)
            loss  = model.criterion(logit, truth)
            dice  = model.metric(logit, truth)

            batch_size = len(index)
            valid_loss += batch_size*np.array(( loss.item(), dice.item(), 0))
            valid_num += batch_size

            prob  = prob [:,:,Y0:Y1, X0:X1]
            truth = truth[:,:,Y0:Y1, X0:X1]
            prob  = F.avg_pool2d(prob,  kernel_size=2, stride=2)
            truth = F.avg_pool2d(truth, kernel_size=2, stride=2)
            predicts.append(prob.data.cpu().numpy())
            truths.append(truth.data.cpu().numpy())

            #assert(valid_num == len(valid_loader.sampler))
            valid_loss  = valid_loss/valid_num

        #--------------------------------------------------------
        predicts_l.append(np.concatenate(predicts).squeeze())
        truths_l.append(np.concatenate(truths).squeeze())
    predicts=np.mean(predicts_l,axis=0)
    truths=np.mean(truths_l,axis=0)
    precision, result, t  = do_kaggle_metric(predicts, truths, threshold)
    res = precision.mean()
    std = precision.std()

    print('Threshold: {}, Metric: {}, std {}'.format(threshold, res,std))
    metric_by_threshold.append((res, threshold))

if TUNE:
 best_metric, best_threshold = max(metric_by_threshold)
else:
 best_metric=0
 best_threshold = 0.45
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
