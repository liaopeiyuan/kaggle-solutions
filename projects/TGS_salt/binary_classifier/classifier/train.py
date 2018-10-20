#from dependencies import *
# Standard imports
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

# Code imports
import model
import pretrainedmodels
import pretrainedmodels.utils as utils

# Image imports
from PIL import Image
from imgaug import augmenters as iaa

# PyTorch imports
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import data_parallel
from tensorboardX import SummaryWriter

from bunny import bunny

def f2_loss(logits, labels):
 __small_value=1e-6
 beta = 2
 batch_size = logits.size()[0]
 p = F.sigmoid(logits)
 l = labels
 num_pos = torch.sum(p, 0) + __small_value
 num_pos_hat = torch.sum(l, 0) + __small_value
 tp = torch.sum(l * p, 0)
 precise = tp / num_pos
 recall = tp / num_pos_hat
 fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + __small_value)
 loss = fs.sum() / batch_size
 return (1 - loss)

# Set all torch tensors as cuda tensors
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Printing options
pd.set_option('display.max_columns', None)

# Seeds for deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

#print(pretrainedmodels.pretrained_settings)

# Neural Network training parameters
initial_learning_rate = 0.01
max_epochs = 500
cyclic_lr_epoch_period = 50
momentum = 0.9
weight_decay = 0.0005
model_name = 'resnet34'
mean = np.array(pretrainedmodels.pretrained_settings[model_name]['imagenet']['mean'])
std = np.array(pretrainedmodels.pretrained_settings[model_name]['imagenet']['std'])
input_size = np.array(pretrainedmodels.pretrained_settings[model_name]['imagenet']['input_size'])[1:]

# Miscellaneous Parameters
depths_filepath = 'data/depths.csv'
train_images_dir = 'data/train'
test_images_dir = 'data/test'
metadata_filepath = 'data/metadata.csv'
checkpoints_dir = 'checkpoints/' + model_name + '/'
log_dir = 'log/' + model_name + '/'
initial_checkpoint  = ''

# Data parameters
number_of_folds = 5
train_params = {'batch_size': 32, 'shuffle': True, 'num_workers': 0, 'pin_memory': False}
validation_params = {'batch_size': 100, 'shuffle': False, 'num_workers': 0, 'pin_memory': False}
test_params = {'batch_size': 144, 'shuffle': False, 'num_workers': 0, 'pin_memory': False}

# Create directories
os.makedirs(name=checkpoints_dir, exist_ok=True)
os.makedirs(name=log_dir, exist_ok=True)

# Create summary writer for tensorboard visualization
summary = SummaryWriter(log_dir)

def test_augment(image):
    x = []
    tensors = []
    x.append(transforms.resize(img=image, size=input_size, interpolation=Image.BILINEAR))
    x.append(transforms.hflip(x[0]))
    x.append(transforms.adjust_brightness(x[0], brightness_factor=0.4*np.random.random() + 0.8))
    x.append(transforms.adjust_hue(x[0], hue_factor=0.2*np.random.random() - 0.1))
    
    for i, img in enumerate(x):
        tensors.append(transforms.to_tensor(img))
        tensors[i] = transforms.normalize(tensors[i], mean, std).cuda()
    
    return tensors

def validation_augment(image):
    x = []
    tensors = []
    x.append(transforms.resize(img=image, size=input_size, interpolation=Image.BILINEAR))
    x.append(transforms.hflip(x[0]))
    x.append(transforms.adjust_brightness(x[0], brightness_factor=0.4*np.random.random() + 0.8))
    x.append(transforms.adjust_hue(x[0], hue_factor=0.2*np.random.random() - 0.1))
    
    for i, img in enumerate(x):
        tensors.append(transforms.to_tensor(img))
        tensors[i] = transforms.normalize(tensors[i], mean, std).cuda()
    
    return tensors

def train_augment(image):
    x = image
    x = transforms.resize(img=x, size=input_size, interpolation=Image.BILINEAR)
    
    # Random horizontal flip
    if np.random.random() >= 0.5:
        x = transforms.hflip(x)
        
    # Brightness, hue or no adjustment
    c = np.random.choice(3)
    if c == 0:
        pass
    elif c == 1:
        x = transforms.adjust_brightness(x, brightness_factor=0.4*np.random.random() + 0.8)
    elif c == 2:
        x = transforms.adjust_hue(x, hue_factor=0.2*np.random.random() - 0.1)
        
    x = transforms.to_tensor(x)
    x = transforms.normalize(x, mean, std)
    
    return x.cuda()

class TGS_Dataset(Dataset):
    def __init__(self, data, augment=train_augment, mode='train'):
        self.data = data
        self.augment = augment
        self.mode = mode        
        self.x = []
        self.y = []
        for i in range(len(self.data)):
            img = Image.open(self.data['file_path_image'][i])
            self.x.append(img)
            fp = img.fp
            img.load()
            if self.mode != 'test':
                img = Image.open(self.data['file_path_mask'][i])
                label = np.array(img)
                fp = img.fp
                img.load()
                if np.sum(label) > 0:
                    self.y.append(1.0)
                else:
                    self.y.append(0.0)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'test':
            return self.augment(self.x[index]), self.data['id'][index]           
        else:
            return self.augment(self.x[index]), torch.tensor(self.y[index]).cuda()


def prepare_data():
    depths = pd.read_csv(depths_filepath)
    
    # Load train metadata
    metadata = {}
    for filename in tqdm(os.listdir(os.path.join(train_images_dir, 'images'))):
        image_filepath = os.path.join(train_images_dir, 'images', filename)
        mask_filepath = os.path.join(train_images_dir, 'masks', filename)
        image_id = filename.split('.')[0]
        depth = depths[depths['id'] == image_id]['z'].values[0]
    
        # calculate salt coverage
        mask = np.array(Image.open(mask_filepath))
        salt_coverage = np.sum(mask > 0) / (mask.shape[0]*mask.shape[1])
        
        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(mask_filepath)
        metadata.setdefault('is_train', []).append(1)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('z', []).append(depth)
        metadata.setdefault('salt_coverage', []).append(salt_coverage)
    
    # Sort by coverage and split in n folds
    data = pd.DataFrame.from_dict(metadata)
    data.sort_values('salt_coverage', inplace=True)
    data['fold'] = (list(range(number_of_folds))*data.shape[0])[:data.shape[0]]       
    
    # Load test metadata
    metadata = {}
    for filename in tqdm(os.listdir(os.path.join(test_images_dir, 'images'))):
        image_filepath = os.path.join(test_images_dir, 'images', filename)
        image_id = filename.split('.')[0]
        depth = depths[depths['id'] == image_id]['z'].values[0]

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(None)
        metadata.setdefault('fold', []).append(None)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('is_train', []).append(0)
        metadata.setdefault('salt_coverage', []).append(None)  
        metadata.setdefault('z', []).append(depth)

    data = data.append(pd.DataFrame.from_dict(metadata), ignore_index=True)
    data.to_csv(metadata_filepath, index=None)
    
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    lr = lr[0]
    return lr

def F_score(logit, label, threshold=0.5):
    prob = torch.sigmoid(logit)
    prob = prob > threshold
    label = label > threshold

    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()

    return torch.tensor([TP, TN, FP, FN]).cuda()

def train():

    # Get train data
    meta = pd.read_csv(metadata_filepath)
    meta_train = meta[meta['is_train'] == 1]  
    test_set = meta[meta['is_train'] == 0].reset_index(drop=True)
    
    print(number_of_folds)
 
    for fold in range(number_of_folds): 
        # Train / validation split
        train_set = meta_train[meta_train['fold'] != fold].reset_index(drop=True)
        validation_set = meta_train[meta_train['fold'] == fold].reset_index(drop=True)
        
        # Generators
        tgs_train = TGS_Dataset(train_set, augment=train_augment, mode='train')
        tgs_validation = TGS_Dataset(validation_set, augment=validation_augment, mode='validation')
        
        train_generator = DataLoader(tgs_train, **train_params)
        validation_generator = DataLoader(tgs_validation, **validation_params)
            
        # Define model and load checkpoint
        net = model.classifier(model_name).cuda()
        if initial_checkpoint != '':
            state = torch.load(checkpoints_dir + initial_checkpoint)
            net.load_state_dict(state['state_dict'])
            
        # Define optimizer and loss
        #optimizer = optim.Adam(net.parameters(), lr=initial_learning_rate,weight_decay=weight_decay)
        optimizer = optim.SGD(net.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=False)
        criterion = nn.BCEWithLogitsLoss().cuda()
        
        # Define cyclic learning rate
        lr_scheduler = lambda x: 0.5 * initial_learning_rate * (1 + np.cos(np.pi * (x % (cyclic_lr_epoch_period*len(train_set))) / (cyclic_lr_epoch_period*len(train_set))))
        
        # Initialize variables
        best_validation_F1 = 0.0        
        images_processed = 0
        
        for epoch in bunny(range(max_epochs)):
            
            # Training process
            torch.cuda.empty_cache()
            net.set_mode('train')
     
            # Initialize variables
            train_loss = 0.0
            train_metric = torch.zeros(4).cuda()
            for input, target in train_generator:

                # Get student logits
                logits = data_parallel(net, input).squeeze()
                
                # loss calculation
                #print("logit ",logits.size())
                #print("target ",target.size())
                loss = criterion(logits, target)

                # Make a step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate metrics and save results
                train_metric += F_score(logits, target)
                train_loss += loss
                    
                # Track number of images being processed for cyclic learning rate
                images_processed += input.shape[0]
                
                # Adjust cyclic learning rate
                adjust_learning_rate(optimizer, lr_scheduler(images_processed))                
                
            # Calculate train metrics
            train_precision = train_metric[0] / (train_metric[0] + train_metric[2] + 1e-12)
            train_recall = train_metric[0] / (train_metric[0] + train_metric[3] + 1e-12)
            train_F1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-12)               
            
            # Validation process
            net.set_mode('validation')
            torch.cuda.empty_cache()
            
            # Initialize variables
            validation_loss = 0.0
            validation_metric = torch.zeros(4).cuda()
            
            with torch.no_grad():
                for input, target in validation_generator:
                    logits = torch.zeros(target.shape).cuda()
                    for tensor in input:
                        logits += data_parallel(net, tensor).squeeze()
                    logits /= len(input)
                    
                    # Metrics calculation
                    validation_metric += F_score(logits, target)
                    validation_loss += criterion(logits, target)
            
            # Calculate train metrics
            validation_precision = validation_metric[0] / (validation_metric[0] + validation_metric[2] + 1e-12)
            validation_recall = validation_metric[0] / (validation_metric[0] + validation_metric[3] + 1e-12)
            validation_F1 = 2 * validation_precision * validation_recall / (validation_precision + validation_recall + 1e-12)            
            
            # Add summaries
            summary.add_scalar('data/learning rate', get_learning_rate(optimizer), epoch)
            
            summary.add_scalars('data/losses', {'train loss': train_params['batch_size']*train_loss/len(train_set)}, epoch)
            summary.add_scalars('data/losses', {'validation loss': validation_params['batch_size']*validation_loss/len(validation_set)}, epoch)
            
            summary.add_scalars('data/F1_scores', {'train F1 score': train_F1}, epoch)
            summary.add_scalars('data/F1_scores', {'validation F1 score': validation_F1}, epoch)
            
            summary.add_scalars('data/precision', {'train precision': train_precision}, epoch)
            summary.add_scalars('data/precision', {'validation precision': validation_precision}, epoch)
            
            summary.add_scalars('data/recall', {'train recall': train_recall}, epoch)
            summary.add_scalars('data/recall', {'validation recall': validation_recall}, epoch)
            
            summary.add_scalars('data/TP', {'train TP': train_metric[0]}, epoch)
            summary.add_scalars('data/TP', {'validation TP': validation_metric[0]}, epoch)
            
            summary.add_scalars('data/TN', {'train TN': train_metric[1]}, epoch)
            summary.add_scalars('data/TN', {'validation TN': validation_metric[1]}, epoch)
            
            summary.add_scalars('data/FP', {'train FP': train_metric[2]}, epoch)
            summary.add_scalars('data/FP', {'validation FP': validation_metric[2]}, epoch)
            
            summary.add_scalars('data/FN', {'train FN': train_metric[3]}, epoch)
            summary.add_scalars('data/FN', {'validation FN': validation_metric[3]}, epoch)
            
            # Save best model
            if best_validation_F1 < validation_F1:
                best_validation_F1 = validation_F1
            state = {'epoch': epoch + 1,
                         'learning_rate': get_learning_rate(optimizer),
                         'validation_LB': best_validation_F1,
                         'state_dict': net.state_dict(),
                         'optimizer' : optimizer.state_dict()}
            torch.save(state, checkpoints_dir + 'fold_' + str(fold) + 'epoch_' + str(epoch) + '_checkpoint.pth')
                
        # Run only one fold for now
        #break

def submission():
    # Gather test data
    meta = pd.read_csv(metadata_filepath)
    meta_test = meta[meta['is_train'] == 0].reset_index(drop=True)
    
    # Generators
    tgs_test = TGS_Dataset(meta_test, augment=test_augment, mode='test')    
    test_generator = DataLoader(tgs_test, **test_params)
    
    # Submission process
    pred_dict = {}
    nets = []
   
     
    checkpoints = [

'fold_0epoch_26_checkpoint_tensor(0.9436).pth',

'fold_0epoch_39_checkpoint_tensor(0.9516).pth',

'fold_0epoch_52_checkpoint_tensor(0.9247).pth',

'fold_1epoch_36_checkpoint_tensor(0.9353).pth',

'fold_1epoch_57_checkpoint_tensor(0.9358).pth',

'fold_1epoch_88_checkpoint_tensor(0.9302).pth',

'fold_2epoch_28_checkpoint_tensor(0.9458).pth',

'fold_2epoch_37_checkpoint_tensor(0.9544).pth',

'fold_2epoch_89_checkpoint_tensor(0.9414).pth',

'fold_3epoch_19_checkpoint_tensor(0.9441).pth',

'fold_3epoch_99_checkpoint_tensor(0.9385).pth',

'fold_3epoch_8_checkpoint_tensor(0.9405).pth',

'fold_4epoch_19_checkpoint_tensor(0.9277).pth',

'fold_4epoch_9_checkpoint_tensor(0.9234).pth',

'fold_4epoch_98_checkpoint_tensor(0.9253).pth',

'fold_5epoch_18_checkpoint_tensor(0.9426).pth',

'fold_5epoch_49_checkpoint_tensor(0.9431).pth',

'fold_5epoch_97_checkpoint_tensor(0.9339).pth',

'fold_6epoch_27_checkpoint_tensor(0.9429).pth',

'fold_6epoch_86_checkpoint_tensor(0.9331).pth',

'fold_6epoch_79_checkpoint_tensor(0.9344).pth',

'fold_7epoch_29_checkpoint_tensor(0.9341).pth',

'fold_7epoch_42_checkpoint_tensor(0.9433).pth',

'fold_7epoch_79_checkpoint_tensor(0.9347).pth',

'fold_8epoch_18_checkpoint_tensor(0.9378).pth',

'fold_8epoch_99_checkpoint_tensor(0.9367).pth',

'fold_8epoch_78_checkpoint_tensor(0.9367).pth',

'fold_9epoch_25_checkpoint_tensor(0.9607).pth',

'fold_9epoch_36_checkpoint_tensor(0.9500).pth',

'fold_9epoch_99_checkpoint_tensor(0.9500).pth'

]

 

    # Pad limits for loss calculation
    #dy0, dy1, dx0, dx1 = compute_center_pad(101, 101, factor=32)    
    ROOT = '/data/ml-arsenal/projects/TGS_salt/binary_classifier/classifier/checkpoints/resnet34/' 
    # Grab training checkpoints
    net = model.classifier(model_name).cuda()
    #net = model.Unet_resnet().cuda()
    state = torch.load(ROOT + 'fold_9epoch_98_checkpoint_tensor(0.9518).pth')
    net.load_state_dict(state['state_dict'])
    net.set_mode('test')
    
    empty = 0
    with torch.no_grad():
        for input, id_name in tqdm(test_generator):
            #print(len(input))
            #print(np.array(input[1]).shape)
            input1 = torch.from_numpy(np.array(input[0])).cuda()
            input2 = torch.from_numpy(np.array(input[1])).cuda()
            
            logit1 = []
            logit2 = []
            for checkpoint in checkpoints:
                state = torch.load(ROOT + checkpoint)
                net.load_state_dict(state['state_dict'])
                net.set_mode('test')
                if type(logit1) is list:
                    logit1 = net(input1)
                    prob1 = F.sigmoid(logit1)
                else: 
                    logit1 = net(input1)
                    prob1 = prob1+F.sigmoid(logit1)
                if type(logit2) is list:
                    logit2 = net(input2) 
                    prob2 = F.sigmoid(logit2)
                else: 
                    logit2 = net(input2)
                    prob2 = prob2+F.sigmoid(logit2)
            
            prob1 = prob1/len(checkpoints)
            prob2 = prob2/len(checkpoints)
            
            prob = (prob1+prob2)/2
            prob = prob.cpu().detach().numpy()
            prob = prob>0.65
            #print(prob) 
            
            for i in range(len(id_name)):
                if prob[i]==1:
                    pred_dict[id_name[i]] = ""
                else:
                    empty = empty+1
                    pred_dict[id_name[i]] = "1 1"
         
        """        
        for input, id_name in tqdm(test_generator, total=len(meta_test)/test_params['batch_size']):
            #print(input)
            input = torch.stack(input).cuda()
            #logits = torch.zeros(input.shape[0], input.shape[2], input.shape[3], input.shape[4]).cuda()
            #for k in range(input.shape[2]):                    
            #    logits[:, k:k+1, :, :] = data_parallel(net, input[:, :, k, :, :])                        
            print(net(input))
            break   
            # TTA validation
            logits[:, 0, :, :] += torch.flip(logits[:, 1, :, :], [2])
            #output = torch.sigmoid((logits[:, 0, :, :]/2)[:, dx0:dx0+101, dy0:dy0+101]) > 0.50
            
            for j in range(logits.shape[0]):
                x = input[j, 0, 0, :, :]*std[0] + mean[0]
                if x.sum() == 0:
                    output[j, :, :] = torch.zeros(101, 101)
                pred_dict[id_name[j]] = rle_encode(output[j, :, :].data.cpu().numpy())            
        """ 
        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv('submission_bc.csv')
        print("Number of empty masks: ", empty)            
            

if __name__ == "__main__":
    prepare_data()
    train()
    #submission()
