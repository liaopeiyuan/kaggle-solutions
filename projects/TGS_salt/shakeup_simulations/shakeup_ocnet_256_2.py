import os
import sys
sys.path.append('../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.SEResnextUnet50_OC_scSE_hyper import SeResNeXt50Unet as Net
import pickle
TUNE = False
SIZE= 0
PAD  = 0
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,
#FOLD=["0","1","2"]

FOLD=[]
for i in range(1,2):
    FOLD.append(str(i))

RANGE=[]
for i in range(1,2):
    test_path = os.path.join('/data/liao_checkpoints/ocnet_256/fold1_stage2')
    print(test_path)
    test_file_list = glob.glob(os.path.join(test_path, '*model.pth'))
    test_file_list = [f.split('/')[-1].split('.pth')[0] for f in test_file_list]
    RANGE.append(test_file_list)

print(FOLD)
print(RANGE)

"""
for i in range(7):
    RANGE.append(np.arange(int(59.5*1000),109*1000,10*1000))
"""
#FOLD=["6","9","0","1","8","7","5","4","3","2"]
#RANGE=[np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(119.5*1000),170*1000,10*1000),np.arange(int(49.5*1000),100*1000,10*1000),np.arange(int(59.5*1000),110*1000,10*1000),np.arange(int(109.5*1000),160*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(99.5*1000),150*1000,10*1000),np.arange(int(89.5*1000),140*1000,10*1000)]
batch_size=15
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
    if height % 256 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 256 - height % 256
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 256 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 256 - width % 256
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img / 255.0
        return torch.from_numpy(img).float()
    else:
        img = img / 255.0
        return torch.from_numpy(img.reshape((1,256,256))).float()


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

        #prob  = prob [:,:,Y0:Y1, X0:X1]
        #truth = truth[:,:,Y0:Y1, X0:X1]
        #prob  = F.avg_pool2d(prob,  kernel_size=2, stride=2)
        #truth = F.avg_pool2d(truth, kernel_size=2, stride=2)
        predicts.append(prob.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())

    assert(valid_num == len(valid_loader.sampler))
    valid_loss  = valid_loss/valid_num
    print(predicts[0].shape)
    print(truths[0].shape)
    #--------------------------------------------------------
    predicts = np.concatenate(predicts).squeeze()
    truths   = np.concatenate(truths).squeeze()
    precision, result, threshold  = do_kaggle_metric(predicts, truths, threshold)
    valid_loss[2] = precision.mean()

    return valid_loss

batch_size = 25
net = Net().cuda()
public_losses=[]
private_losses=[]
i=-1
results={}
m=0
for fold in FOLD:
 for step in RANGE[m]:
   seeds = {}
   SIZE = 202
   bsize = 22
   #ResNet34_OHEM00079500_model.pth 
   initial_checkpoint = '/data/liao_checkpoints/ocnet_256/fold1_stage2/'+step+'.pth'
   model = "OCnet256_"+'fold'+fold+step
   from tqdm import tqdm
   if initial_checkpoint is not None:
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
   pub_lbs=[]
   priv_lbs=[]
   public_losses=[]
   private_losses=[]

   for seed in range(0,50000,500):
    net.set_mode('valid')
    public_losses=[]
    private_losses=[]
    #pub_lbs=[]
    #priv_lbs=[]

    valid_dataset = TGSDataset('simulated/sim_private'+str(seed)+'_2668_ne', valid_augment, 'train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = bsize,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)
    private_loss = validation(net, valid_loader, 0.5)
    private_losses.append(private_loss)
    priv_lbs.append(private_loss[2])

    pickle.dump( private_losses, open( "private_losses.p", "wb" ) )
    print("Private Dataset | Seed: {}, dice: {}, LB: {}".format(seed,round(private_loss[1],5),round(private_loss[2],5)))

    valid_dataset = TGSDataset('simulated/sim_public'+str(seed)+'_1332_ne', valid_augment, 'train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = bsize,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)
    public_loss = validation(net, valid_loader, 0.5)
    public_losses.append(public_loss)
    pub_lbs.append(public_loss[2])

    pickle.dump( public_losses, open( "public_losses.p", "wb" ) )
    print("Public Dataset | Seed: {}, dice: {}, LB: {}".format(seed,round(public_loss[1],5),round(public_loss[2],5)))

    seeds[seed]=[public_losses,private_losses]
    
    print("Observed difference: {}".format(public_loss[2]-private_loss[2]))
    
    pickle.dump( seeds, open( model+"_simulation_results.p", "wb" ) )
    
    pub=np.array(pub_lbs)
    priv=np.array(priv_lbs)
    print("Mean Difference: {}".format(np.mean(pub-priv)))
    print("Expected shakeup (+-): {}".format(np.std(pub-priv)))
 m=m+1
 results[model]=seeds
 pickle.dump( results, open( "simulation_2.p", "wb" ) )
