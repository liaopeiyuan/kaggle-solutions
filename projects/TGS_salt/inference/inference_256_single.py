import os
import sys
sys.path.append('../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net
import pickle
TUNE = False
SIZE = 256
PAD  = 256
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,
#FOLD=["0","1","2"]

EVAL = True
FOLD=[]
for i in range(10):
    FOLD.append(str(i))

RANGE=[]
for i in range(10):
    RANGE.append(np.arange(int(9.5*1000),120*1000,10*1000))
 
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

    #img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img / 255.0
        return torch.from_numpy(img).float()
    else:
        image = img / 255.0
        image, _ = do_resize2(image,image, SIZE, SIZE)
        image, _ = do_center_pad_to_factor2(image,image, factor=256)
        return torch.from_numpy(image.reshape((1,256,256))).float()


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



valid_dataset = TGSDataset('list_test_18000', valid_augment, 'test')
valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = batch_size,
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
        #image, _ = do_resize2(image,image, SIZE, SIZE)
        #image, _ = do_center_pad_to_factor2(image,image, factor=256)
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
pickle.dump( test_dataset, open( "test_dataset.p", "wb" ) )
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

all_predictions_stacked=[]
m=0
i=int(167.5*1000)
if i > 59.5*1000:
    stage="256"
else:
    stage=""
initial_checkpoint = '/home/liaop20/data/salt/checkpoints/list_train6_3600_ne_balanced/ResNet34_res_25600098000_model.pth'
model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
model.set_mode('test')
model.eval()
all_predictions = []

#EVAL = True
if EVAL:
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
 else:
    all_predictions_stacked= all_predictions_stacked+np.vstack(all_predictions)[:, 0, :, :]

if EVAL:
 pickle.dump( all_predictions_stacked[0:3000,:,:], open( "256_sigmoids1.p", "wb" ) )
 pickle.dump( all_predictions_stacked[3000:6000,:,:], open( "256_sigmoids2.p", "wb" ) )
 pickle.dump( all_predictions_stacked[6000:9000,:,:], open( "256_sigmoids3.p", "wb" ) )
 pickle.dump( all_predictions_stacked[9000:12000,:,:], open( "256_sigmoids4.p", "wb" ) )
 pickle.dump( all_predictions_stacked[12000:15000,:,:], open( "256_sigmoids5.p", "wb" ) )
 pickle.dump( all_predictions_stacked[15000:18000,:,:], open( "256_sigmoids6.p", "wb" ) )
m=m+1

pred1  = pickle.load( open( "256_sigmoids1.p", "rb" ) )
pred2  = pickle.load( open( "256_sigmoids2.p", "rb" ) )
pred3  = pickle.load( open( "256_sigmoids3.p", "rb" ) )
pred4  = pickle.load( open( "256_sigmoids4.p", "rb" ) )
pred5  = pickle.load( open( "256_sigmoids5.p", "rb" ) )
pred6  = pickle.load( open( "256_sigmoids6.p", "rb" ) )

all_predictions_stacked= np.vstack((pred1,pred2,pred3,pred4,pred5,pred6))

del pred1,pred2,pred3,pred4,pred5,pred6
"""
import codecs, json 

b = all_predictions_stacked.tolist() # nested lists with same data, indices
file_path = "/256_sigmoids.json" ## your path variable
json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
all_predictions_stacked = np.array(b_new)
"""
print(all_predictions_stacked.shape)




#all_predictions_stacked = all_predictions_stacked[:, 27:256 - 27, 27:256 - 27]



print(all_predictions_stacked.shape)

images=[]
for i in tqdm(range(18000)):
   img = cv2.resize(all_predictions_stacked[i,:,:],(101,101))
   images.append(img)

all_predictions_stacked =np.array(images)
print(all_predictions_stacked.shape)
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
submit.to_csv(RESULT+'/single_snap.csv', index = False)
