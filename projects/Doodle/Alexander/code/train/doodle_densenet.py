#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0,1,2,3' #'3,2,1,0'
import sys
sys.path.append("..")
from common import *
from data   import *
from imgaug import augmenters as iaa

##----------------------------------------
from model_densenet201 import *
# from model_resnet34 import *
import gc


# In[2]:
#net = Net().to('cuda:0')
#print(next(net.parameters()).is_cuda)

#net = nn.DataParallel(net)
#net.cuda()
#print(next(net.parameters()).is_cuda)
#print(net.weights)
#print(type(net))

FILE_NAME = '100k'
SIZE = 144

# In[3]:

# each rgb image, same stroke has same color
def drawing_to_image_with_color_stroke_multi_channel(drawing, H, W, seq, channels):
    cts = int(channels / 3)

    point = []
    time = []
    for t, (x, y) in enumerate(drawing):
        point.append(np.array((x, y), np.float32).T)
        time.append(np.full(len(x), t))

    point = np.concatenate(point).astype(np.float32)
    time = np.concatenate(time).astype(np.int32)
    T = time.max() + 1
    image_all = np.full((H, W, 3), 0, np.uint8)

    colors = plt.cm.jet(np.arange(1, 1 + T) / (T*1.0))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    for ct in range(cts):
        image_tmp = np.full((H, W, 3), 0, np.uint8)
        x_max = point[:, 0].max()
        x_min = point[:, 0].min()
        y_max = point[:, 1].max()
        y_min = point[:, 1].min()
        w = x_max - x_min
        h = y_max - y_min
        # print(w,h)

        s = max(w, h)
        norm_point = (point - [x_min, y_min]) / s
        norm_point = (norm_point - [w / s * 0.5, h / s * 0.5]) * max(W, H) * 0.85
        norm_point = np.floor(norm_point + [W / 2, H / 2]).astype(np.int32)

        # draw
        for t in range(int(ct * T / cts), int((ct + 1) * T / cts)):

            color = colors[t]
            color = [int(color[2]), int(color[1]), int(color[0])]

            p = norm_point[time == t]
            x, y = p.T
            image_tmp[y, x] = 255
            N = len(p)
            for i in range(N - 1):
                x0, y0 = p[i]
                x1, y1 = p[i + 1]
                cv2.line(image_tmp, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

        if ct == 0:
            image_all = image_tmp
        else:
            image_all = np.concatenate((image_all, image_tmp), axis=2)

    image_all = seq.augment_image(image_all)

    return image_all

def mixup_data(x, y, alpha=4.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# In[4]:


def valid_augment(drawing, label, index):
#     image = drawing_to_image_with_color_v2(drawing, 96, 96)
    seq = iaa.Sequential([
    iaa.Crop(percent=(0.05, 0.05, \
                      0.05, 0.05), keep_size=True)
    ])
    #image = drawing_to_image_with_color_stroke_multi_channel(drawing, SIZE, SIZE, seq, 3)
    image = drawing_to_image_with_color_aug(drawing, SIZE, SIZE, seq)
    return image, label, None


def train_augment(drawing, label, index):
    up_rand = np.random.random()
    right_rand = np.random.random()
    percent_crop = 0.1
    seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(up_rand*percent_crop, right_rand*percent_crop, \
                      (1-up_rand)*percent_crop, (1-right_rand)*percent_crop), keep_size=True)
    ])
    
    #image = drawing_to_image_with_color_stroke_multi_channel(drawing, SIZE, SIZE, seq, 3)
    image = drawing_to_image_with_color_aug(drawing, SIZE, SIZE, seq)
#     image = drawing_to_image_with_color_v2(drawing, 96, 96)
    return image, label, None


# In[5]:


### training ##############################################################

def do_valid( net, valid_loader, criterion ):

    valid_num  = 0
    probs    = []
    truths   = []
    losses   = []
    corrects = []

    for input, truth, cache in valid_loader:
        input = input.to('cuda:0')
        truth = truth.to('cuda:0')

        with torch.no_grad():
            logit   = net(input)#data_parallel(net,input)#net(input)
            prob    = F.softmax(logit,1)

            loss    = criterion(logit, truth, False)
            correct = metric(logit, truth, False)

        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())
        losses.append(loss.data.cpu().numpy())
        corrects.append(correct.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())


    assert(valid_num == len(valid_loader.sampler))
    #------------------------------------------------------
    prob    = np.concatenate(probs)
    correct = np.concatenate(corrects)
    truth   = np.concatenate(truths).astype(np.int32).reshape(-1,1)
    loss    = np.concatenate(losses)


    #---
    #top = np.argsort(-predict,1)[:,:3]

    loss    = loss.mean()
    correct = correct.mean(0)

    top = [correct[0], correct[0]+correct[1], correct[0]+correct[1]+correct[2]]
    precision = correct[0]/1 + correct[1]/2 + correct[2]/3

    #----
    valid_loss = np.array([
        loss, top[0], top[2], precision
    ])

    return valid_loss


# In[6]:


fold    = 0
out_dir =     '../../densenet'
initial_checkpoint = '/rscratch/xuanyu/KAIL/Kaggle_Doddle_Rank1/Alexander/densenet/checkpoint/00566000_model.pth'#'densenet201.pth'#None #\
        #'../../output/backup/873_crop.pth'

pretrain_file = None

batch_size = 512
epoch = 20
num_iters   = epoch * 340 * 100000 // batch_size

#     schduler  = NullScheduler(lr=0.01)
schduler = DecayScheduler(base_lr=0.00025, decay=0.1, step=num_iters/2)
iter_save_interval = 2000
criterion          = softmax_cross_entropy_criterion


## setup  -----------------------------------------------------------------------------
os.makedirs(out_dir +'/checkpoint', exist_ok=True)
os.makedirs(out_dir +'/train', exist_ok=True)
os.makedirs(out_dir +'/backup', exist_ok=True)
#     backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

log = Logger()
log.open(out_dir+'/log.train_r50_add_crop.txt',mode='a')
log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
log.write('\tSEED         = %u\n' % SEED)
log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
log.write('\t__file__     = %s\n' % FILE_NAME)
log.write('\tout_dir      = %s\n' % out_dir)
log.write('\n')
log.write('\t<additional comments>\n')
log.write('\t  ... xxx baseline  ... \n')
log.write('\n')


## dataset ----------------------------------------
log.write('** dataset setting **\n')

train_dataset = DoodleDataset('train', 'train_0', train_augment)
train_loader  = DataLoader(
                    train_dataset,
                    #sampler     = FixLengthRandomSamplerWithProbability(train_dataset, probability),
                    #sampler     = FixLengthRandomSampler(train_dataset),
                    #sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
                    sampler     = RandomSampler(train_dataset),
                    batch_size  = batch_size,
                    num_workers = 15,
                    drop_last   = True,
                    pin_memory  = False,
                    collate_fn  = null_collate)

valid_dataset = DoodleDataset('valid', 'valid_0',  valid_augment)
valid_loader  = DataLoader(
                    valid_dataset,
                    #sampler     = SequentialSampler(valid_dataset),
                    sampler     = RandomSampler(valid_dataset),
                    batch_size  = batch_size,
                    num_workers = 15,
                    drop_last   = False,
                    pin_memory  = True,
                    collate_fn  = null_collate)


assert(len(train_dataset)>=batch_size)
log.write('batch_size = %d\n'%(batch_size))
log.write('train_dataset : \n%s\n'%(train_dataset))
log.write('valid_dataset : \n%s\n'%(valid_dataset))
log.write('\n')

## net ----------------------------------------
log.write('** net setting **\n')
#net = Net().cuda()
net = Net().to('cuda:0')
net = nn.DataParallel(net)
net.cuda()
#cudnn.benchmark = True

if initial_checkpoint is not None:
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
#net = nn.DataParallel(net)
#net.cuda()


# In[ ]:


log.write('%s\n'%(type(net)))
log.write('criterion=%s\n'%criterion)
log.write('\n')


## optimiser ----------------------------------
if 0: ##freeze
    for p in net.resnet.parameters(): p.requires_grad = False
    for p in net.encoder1.parameters(): p.requires_grad = False
    for p in net.encoder2.parameters(): p.requires_grad = False
    for p in net.encoder3.parameters(): p.requires_grad = False
    for p in net.encoder4.parameters(): p.requires_grad = False
    pass

#net.set_mode('train',is_freeze_bn=True)
#-----------------------------------------------


optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=schduler.get_rate(0), momentum=0.9, weight_decay=0.0001)


iter_smooth = 20
iter_log    = 50
iter_valid  = 2500
iter_save   = [0, num_iters-1]               + list(range(0, num_iters, iter_save_interval))#1*1000

start_iter = 566000
start_epoch= 0
rate       = 0
if initial_checkpoint is not None:
#     initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
#     checkpoint  = torch.load(initial_optimizer)
#     start_iter  = checkpoint['iter' ]
#     start_epoch = checkpoint['epoch']

    #rate = get_learning_rate(optimizer)  #load all except learning rate
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #adjust_learning_rate(optimizer, rate)
    pass




log.write('schduler\n  %s\n'%(schduler))
log.write('\n')

## start training here! ##############################################
log.write('** start training here! **\n')
log.write('                    |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
log.write('rate   iter  epoch  | loss   acc-1  acc-3   lb       | loss   acc-1  acc-3   lb      |  time   \n')
log.write('----------------------------------------------------------------------------------------------------\n')


train_loss   = np.zeros(6,np.float32)
valid_loss   = np.zeros(6,np.float32)
batch_loss   = np.zeros(6,np.float32)
iter = 0
i    = 0
last_max_lb   = -1


start = timer()
while  iter<num_iters:
    sum_train_loss = np.zeros(6,np.float32)
    sum = 0


    optimizer.zero_grad()
    for input, truth, cache in train_loader:

        len_train_dataset = len(train_dataset)
        batch_size = len(cache)
        iter = i + start_iter
        epoch = (iter-start_iter)*batch_size/len_train_dataset + start_epoch
        num_samples = epoch*len_train_dataset


        if (iter % iter_valid==0) and (iter!=0):
            #net.set_mode('valid')
            net.eval()
            valid_loss = do_valid(net, valid_loader, criterion)
            net.train()

            ##--------
            # lb    = valid_loss[7]
            # loss  = valid_loss[0] + valid_loss[4]
            # last_max_lb = max(last_max_lb,lb)
            # if last_max_lb-lb<0.005:
            #     iter_save += [iter,]
            # if loss-last_min_loss<0.005:
            #     iter_save += [iter,]

            asterisk = '*' if iter in iter_save else ' '
            ##--------

            print('\r',end='',flush=True)
            log.write('%0.4f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (                     rate, iter/1000, epoch,
                     valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],asterisk,
                     train_loss[0], train_loss[1], train_loss[2], train_loss[3],
                     time_to_str((timer() - start),'min'))
            )
            log.write('\n')
            time.sleep(0.01)

        #if 0:
        if iter in iter_save:
            torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
            torch.save({
                #'optimizer': optimizer.state_dict(),
                'iter'     : iter,
                'epoch'    : epoch,
            }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
            pass




        # learning rate schduler -------------
        lr = schduler.get_rate(iter)
        if lr<0 : break
        adjust_learning_rate(optimizer, lr)
        rate = get_learning_rate(optimizer)



        # one iteration update  -------------
        #net.set_mode('train',is_freeze_bn=True)
        #net.set_mode('train')
        net.train()
        #input = input.to('cuda:0')
        #truth = truth.to('cuda:0')
        
        inputs, targets_a, targets_b, lam = mixup_data(input, truth, 0.05, True)
        #print(next(net.parameters()).is_cuda) 
        logit = net(inputs.to('cuda:0'))
        #logit = net(input)#data_parallel(net,input) #net(input)
        del input
        targets_a = targets_a.to('cuda:0') 
        loss = mixup_criterion(criterion, logit, targets_a, targets_b.to('cuda:0'), lam)
        precision, top = metric(logit, targets_a)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        del logit, targets_a, targets_b
        gc.collect()
        
        #precision, top = metric(logit, targets_a)
        
        #with torch.no_grad():
        #    input = input.cuda()
        #    truth = truth.cuda()
        #    logit = data_parallel(net, input)
        #loss  = criterion(logit, truth)
        #precision, top = metric(logit, truth)


        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()
        #torch.nn.utils.clip_grad_norm(net.parameters(), 1)


        # print statistics  ------------
        batch_loss[:4] = np.array(( loss.item(), top[0].item(), top[2].item(), precision.item(),))
        sum_train_loss += batch_loss
        sum += 1
        if iter%iter_smooth == 0:
            train_loss = sum_train_loss/sum
            sum_train_loss = np.zeros(6,np.float32)
            sum = 0


        print('\r',end='',flush=True)
        print('%0.4f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (                     rate, iter/1000, epoch,
                     valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3],' ',
                     batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3],
                     time_to_str((timer() - start),'min'))
        , end='',flush=True)
        i=i+1



    pass  #-- end of one data loader --
pass #-- end of all iterations --


if 1: #save last
    torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
    torch.save({
        'optimizer': optimizer.state_dict(),
        'iter'     : i,
        'epoch'    : epoch,
    }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

log.write('\n')


# In[ ]:


input.shape


# In[ ]:




