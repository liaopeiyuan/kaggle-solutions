import os
import sys
sys.path.append('../../')

from dependencies import *
from settings import *
from reproducibility import *
from models.TGS_salt.Unet34_scSE_hyper import Unet_scSE_hyper as Net

SIZE = 128
FACTOR = 128
FOLD = 1
ne = ""
initial_checkpoint = None#'/home/liaop20/data/salt/checkpoints/list_train'+str(FOLD)+'_3600/'#None#'/home/liaop20/data/salt/checkpoints/list_train6_3600_ne_balanced/ResNet34_25600151000_model.pth'#None
MODEL = "ResNet34_"
OHEM = "all_128"
PAD  = 0
Y0, Y1, X0, X1 = PAD,PAD+SIZE,PAD,PAD+SIZE,

def time_to_str(time, str):
    #if str == 'min':
    #	    return str(round(float(time)/60,5))+" min(s)"
    return round(time,4)

#TODO: Instead of directly printing to stdout, copy it into a txt file
class Logger():
    def __init__(self,name=MODEL+OHEM+ne, fold=FOLD):
        super().__init__()
        self.fold=str(fold)
        self.model=name
        #if OHEM != "OHEM":
        #    self.model=MODEL+ne[ne.find("_")+1:]
        self.file = open(self.fold+self.model+"_log.txt","w+")
        self.file.close()
    def write(self, str):
        print(str)
        self.file = open(self.fold+self.model+"_log.txt","a+")
        self.file.write(str)
        self.file.close()
    def write2(self, str):
        print(str, end='',flush=True)
        self.file = open(self.fold+self.model+"_log.txt","a+")
        self.file.write(str)
        self.file.close()
    def stop():
        self.file.close()

def valid_augment(image,mask,index):
    cache = Struct(image = image.copy(), mask = mask.copy())
    image, mask = do_resize2(image, mask, SIZE, SIZE)
    image, mask = do_center_pad_to_factor2(image, mask, factor = FACTOR)
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
    image, mask = do_center_pad_to_factor2(image, mask, factor = FACTOR)
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
            loss  = net.focal_loss(logit, truth, 1.0, 0.5, 0.25) + net.criterion(logit, truth)
            dice  = net.metric(logit, truth)

        batch_size = len(index)
        valid_loss += batch_size*np.array(( loss.item(), dice.item(), 0))
        valid_num += batch_size

        prob  = prob [:,:,Y0:Y1, X0:X1]
        truth = truth[:,:,Y0:Y1, X0:X1]
        #prob  = F.avg_pool2d(prob,  kernel_size=2, stride=2)
        #truth = F.avg_pool2d(truth, kernel_size=2, stride=2)
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

def train(initial_checkpoint):

    ## setup  -----------------
    os.makedirs(CHECKPOINTS +'/checkpoint', exist_ok=True)
    os.makedirs(CHECKPOINTS +'/train', exist_ok=True)
    os.makedirs(CHECKPOINTS +'/backup', exist_ok=True)

    log = Logger()
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % CODE)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tRESULT      = %s\n' % CHECKPOINTS)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... \n')
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('Configuring dataset...\n')
    batch_size = 16

    train_dataset = TGSDataset('list_train'+str(FOLD)+'_3600'+ne, train_augment, 'train')
    os.makedirs(CHECKPOINTS +'/list_train'+str(FOLD)+'_3600'+ne, exist_ok=True)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler     = RandomSampler(train_dataset),
                        #sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    valid_dataset = TGSDataset('list_valid'+str(FOLD)+'_400'+ne, valid_augment, 'train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset.split = %s\n'%(train_dataset.split))
    log.write('valid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\n')

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
    log.write('Configuring neural network...\n')
    net = Net().cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write("The net is an instance of {}.".format(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    num_iters   = 300  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,500))#1*1000

    FREEZE=False
    #------------------------------------------------------
    if FREEZE: ##freeze
        for p in net.feature_net.parameters():
            p.requires_grad = False
    #------------------------------------------------------
    
    scheduler = lambda x: (0.01/2)*(np.cos(PI*(np.mod(x-1,300*1000/30)/(300*1000/30)))+1)
    #log.write(scheduler(1))
    #log.write(scheduler(5000))
    #log.write(scheduler(10001))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=0.0001)
    
    start_iter = 0
    start_epoch= 0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)
        pass


    ## start training here! ##############################################
    log.write('Start training...\n')

    log.write(' rate    iter   epoch   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

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

                log.write2('\r')
                log.write('%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s \n' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str((timer() - start),'min')))
                time.sleep(0.01)

            if iter in iter_save:
                torch.save(net.state_dict(),CHECKPOINTS+"/"+train_dataset.split+'/'+MODEL+OHEM+'%08d_model.pth'%(iter))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, CHECKPOINTS+"/"+train_dataset.split+'/'+MODEL+OHEM+'%08d_optimizer.pth'%(iter))
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

            if OHEM == "OHEM":
                loss = net.focal_loss(logit, truth, 1.0, 0.5, 0.25) + net.criterion(logit, truth)
            else:
                loss  = net.criterion(logit, truth)

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



            log.write2('\r%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s ' % (\
                         rate, iter/1000, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_loss[0], train_loss[1],
                         batch_loss[0], batch_loss[1],
                         time_to_str((timer() - start), 'min')))
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')


if __name__ == '__main__':
    print("Training U-Net with hypercolumn concatenation and spatial/channel-wise excitation...")
    train(initial_checkpoint)
    print('\tFinished!')
