import os
import sys
sys.path.append('../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net

SIZE = 101
PAD  = 27
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,

def time_to_str(time, str):
    #if str == 'min':
    #	    return str(round(float(time)/60,5))+" min(s)"
    return time

#TODO: Instead of directly printing to stdout, copy it into a txt file
class Logger():
    def __init__(self,path=None):
        super().__init__()
        self.path=path
    def write(str):
        print(str)

def valid_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    image, mask = do_resize2(image, mask, SIZE, SIZE)
    image, mask = do_center_pad_to_factor2(image, mask)
    return image,mask,index,cache




def train_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())

    if np.random.rand() < 0.5:
         image, mask = do_horizontal_flip2(image, mask)
         pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c==0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2) #0.125

        if c==1:
            image, mask = do_horizontal_shear2( image, mask, dx=np.random.uniform(-0.07,0.07) )
            pass

        if c==2:
            image, mask = do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0,15))  #10

        if c==3:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0,0.15))#0.10
            pass


    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = do_brightness_shift(image,np.random.uniform(-0.1,+0.1))
        if c==1:
            image = do_brightness_multiply(image,np.random.uniform(1-0.08,1+0.08))
        if c==2:
            image = do_gamma(image,np.random.uniform(1-0.08,1+0.08))
        # if c==1:
        #     image = do_invert_intensity(image)

    image, mask = do_resize2(image, mask, SIZE, SIZE)
    image, mask = do_center_pad_to_factor2(image, mask)
    #print(image.shape)
    return image,mask,index,cache

def validation( net, valid_loader ):

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
    precision, result, threshold  = do_kaggle_metric(predicts, truths)
    valid_loss[2] = precision.mean()

    return valid_loss

def train():

    initial_checkpoint = None
       #'checkpoint/00048500_model.pth'\
    #    None  #'/root/share/project/kaggle/tgs/results/resnet34-resize128-focus/fold0-1a/checkpoint/00003500_model.pth'

    ## setup  -----------------
    os.makedirs(CHECKPOINTS +'/checkpoint', exist_ok=True)
    os.makedirs(CHECKPOINTS +'/train', exist_ok=True)
    os.makedirs(CHECKPOINTS +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, RESULT +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    #log.open(RESULT+'/log.train.txt',mode='a')
    print('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    print('\tSEED         = %u\n' % SEED)
    print('\tPROJECT_PATH = %s\n' % CODE)
    print('\t__file__     = %s\n' % __file__)
    print('\tRESULT      = %s\n' % CHECKPOINTS)
    print('\n')
    print('\t<additional comments>\n')
    print('\t  ... \n')
    print('\n')


    ## dataset ----------------------------------------
    print('Configuring dataset...\n')
    batch_size = 16

    train_dataset = TGSDataset('list_train8_3600', train_augment, 'train')
    os.makedirs(CHECKPOINTS +'/list_train8_3600', exist_ok=True)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler     = RandomSampler(train_dataset),
                        #sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    valid_dataset = TGSDataset('list_valid8_400', valid_augment, 'train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(train_dataset)>=batch_size)
    print('batch_size = %d\n'%(batch_size))
    print('train_dataset.split = %s\n'%(train_dataset.split))
    print('valid_dataset.split = %s\n'%(valid_dataset.split))
    print('\n')

    #debug
    if 0: #debug  ##-------------------------------

        for input, truth, index, cache in train_loader:
            images = input.cpu().data.numpy().squeeze()
            masks  = truth.cpu().data.numpy().squeeze()
            batch_size = len(index)
            for b in range(batch_size):
                image = images[b]*255
                image = np.dstack([image,image,image])

                mask = masks[b]

                image_show('image',image,resize=2)
                image_show_norm('mask', mask, max=1,resize=2)


                overlay0 = draw_mask_overlay(mask, image, color=[0,0,255])
                overlay0 = draw_mask_to_contour_overlay(mask, overlay0, 2, color=[0,0,255])


                image_show('overlay0',overlay0,resize=2)
                cv2.waitKey(0)
    #--------------------------------------


    ## net ----------------------------------------
    print('Configuring neural network...\n')
    net = Net().cuda()

    if initial_checkpoint is not None:
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    print("The net is an instance of {}.".format(type(net)))
    print('\n')



    ## optimiser ----------------------------------
    num_iters   = 300  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    epoch_save   = np.arange(0,1500,10)#[0, num_iters-1]\
                   #+ list(range(0,num_iters,500))#1*1000

    FREEZE=False
    #------------------------------------------------------
    if FREEZE: ##freeze
        for p in net.feature_net.parameters():
            p.requires_grad = False
    
    #from cls import CyclicLR
    #net.set_mode('train',is_freeze_bn=True)
    #------------------------------------------------------
    scheduler = lambda x: (0.009/2)*(np.cos(PI*(np.mod(x-1,int(11.25*1000))/(int(11.25*1000))))+1)+0.001
    print(scheduler(1))
    print(scheduler(5000))
    print(scheduler(10001))
    #scheduler = CyclicLR(base_lr=0.01, max_lr=0.01, step_size=10000, gamma=1., scale_fn=clr_fn, scale_mode='iterations')
    #schduler = None #StepLR([ (0, 0.01),  (200, 0.001)])
    #base_params = list(map(id, net.resnet.parameters())) 
    #decode_params = filter(lambda p: id(p) not in base_params, net.parameters()) 
    #params = [ {"params": decode_params, "lr": 0.01}, 
    #    {"params": net.resnet.parameters(), "lr": 0.005}, ] 
    #optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=0.0001)
    #scheduler = CyclicLR(optimizer,base_lr=0.01, max_lr=0.01, step_size=10000, gamma=1., scale_fn=clr_fn, scale_mode='iterations')
    #scheduler= CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size=10000, gamma=0.99, mode='cos_anneal')
    
    start_iter = 0
    start_epoch= 0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        #optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)
        pass


    ## start training here! ##############################################
    print('Start training...\n')

    #print(' samples_per_epoch = %d\n\n'%len(train_dataset))
    print(' rate    iter   epoch   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    print('-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    valid_loss  = np.zeros(6,np.float32)
    batch_loss  = np.zeros(6,np.float32)
    rate = 0
    iter = 0
    i    = 0

    start = timer()
    while  iter<num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0

        optimizer.zero_grad()
        for input, truth, index, cache in train_loader:

            if 0: #debug  ##-------------------------------

                image = input.cpu().data.numpy().squeeze()
                mask  = truth.cpu().data.numpy().squeeze()


                batch_size = len(index)
                for b in range(batch_size):
                    image_show_norm('image',image[b],max=1,resize=2)
                    image_show_norm('mask', mask[b], max=1,resize=2)
                    cv2.waitKey(0)
            #--------------------------------------

            len_train_dataset = len(train_dataset)
            batch_size = len(index)
            iter = i + start_iter
            epoch = (iter-start_iter)*batch_size/len_train_dataset + start_epoch
            num_samples = epoch*len_train_dataset


            if iter % iter_valid==0:
                net.set_mode('valid')
                valid_loss = validation(net, valid_loader)
                net.set_mode('train')

                print('\r',end='',flush=True)
                print('%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s \n' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str((timer() - start),'min')))
                time.sleep(0.01)

            #if 1:
            if round(epoch,1) == 0 or round(epoch,1) == 1 or round(epoch,1)+0.1 in epoch_save:
                torch.save(net.state_dict(),CHECKPOINTS+"/"+train_dataset.split+'/%08d_model.pth'%(int(round(epoch,1)+0.1)))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, CHECKPOINTS+"/"+train_dataset.split+'/%08d_optimizer.pth'%(int(round(epoch,1)+0.1)))
                pass



            # learning rate schduler -------------
            if scheduler is not None:
                #scheduler.batch_step()
                lr = scheduler(iter)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)
            #rate = 0.01

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)
            net.set_mode('train')

            input = input.cuda()
            truth = truth.cuda()

            logit = data_parallel(net,input) #net(input)
            loss  = net.criterion(logit, truth)
            #loss = torch.nn.BCEWithLogitsLoss(logit,truth)
            dice  = net.metric(logit, truth)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #torch.nn.utils.clip_grad_norm(net.parameters(), 1)


            # print statistics  ------------
            batch_loss = np.array((
                           loss.item(),
                           dice.item(),
                           0, 0, 0, 0,
                         ))
            sum_train_loss += batch_loss
            sum += 1
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0


            print('\r%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s ' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str((timer() - start), 'min')), end='',flush=True)
            i=i+1


            #<debug> ===================================================================
            if 0:
            #if iter%200==0:
                #voxel, aux, query, link, truth, cache = make_valid_batch(valid_dataset.dataset, batch_size=2)

                net.set_mode('test')#
                with torch.no_grad():
                    logit = net(input)
                    prob  = F.sigmoid(logit)
                    loss  = net.criterion(logit, truth)
                    dice  = net.metric(logit, truth)

                    if 0:
                        loss  = net.criterion(logit, truth)
                        accuracy,hit_rate,precision_rate = net.metric(logit, truth)
                        valid_loss[0] = loss.item()
                        valid_loss[1] = accuracy.item()
                        valid_loss[2] = hit_rate.item()
                        valid_loss[3] = precision_rate.item()



                #show only b in batch ---
                b = 1
                prob   = prob.data.cpu().numpy()[b].squeeze()
                truth  = truth.data.cpu().numpy()[b].squeeze()
                input  = input.data.cpu().numpy()[b].squeeze()

                all = np.hstack([input,truth,prob])
                image_show_norm('all',all,max=1,resize=3)
                cv2.waitKey(100)

                net.set_mode('train')
            #<debug> ===================================================================


        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1: #save last
        torch.save(net.state_dict(),CHECKPOINTS +'/checkpoint/'+train_dataset.split+'/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, CHECKPOINTS +'/checkpoint/'+train_dataset.split+'/%d_optimizer.pth'%(i))

    print('\n')


if __name__ == '__main__':
    print("Training U-Net with hypercolumn concatenation and spatial/channel-wise excitation...")
    train()
    print('\tFinished!')
