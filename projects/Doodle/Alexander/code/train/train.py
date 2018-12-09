import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0' #'3,2,1,0'

from common import *
from data   import *


##----------------------------------------
from model32_resnet34 import *



def valid_augment(drawing, label, index):
    cache = Struct(drawing = drawing.copy(), label = label, index=index)
    image = drawing_to_image(drawing, 64, 64)
    return image, label, cache


def train_augment(drawing, label, index):
    cache = Struct(drawing = drawing.copy(), label = label, index=index)
    ## <todo> augmentation ....
    image = drawing_to_image(drawing, 64, 64)
    return image, label, cache





### training ##############################################################

def do_valid( net, valid_loader, criterion ):

    valid_num  = 0
    probs    = []
    truths   = []
    losses   = []
    corrects = []

    for input, truth, cache in valid_loader:
        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit   = net(input)
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




def run_train():

    fold    = 0
    out_dir = \
        '../../output'
    initial_checkpoint = \
        None

    pretrain_file = \
        '../../models/resnet34-333f7ec4.pth'


    schduler  = NullScheduler(lr=0.01)
    iter_save_interval = 2000
    criterion          = softmax_cross_entropy_criterion


    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 128  #16 #32

    train_dataset = DoodleDataset('train', 'train_use', train_augment)
    train_loader  = DataLoader(
                        train_dataset,
                        #sampler     = FixLengthRandomSamplerWithProbability(train_dataset, probability),
                        #sampler     = FixLengthRandomSampler(train_dataset),
                        #sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
                        sampler     = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 2,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    valid_dataset = DoodleDataset('train', 'valid',  valid_augment)
    valid_loader  = DataLoader(
                        valid_dataset,
                        #sampler     = SequentialSampler(valid_dataset),
                        sampler     = RandomSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True,
                        collate_fn  = null_collate)


    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain(pretrain_file)
        #net = load_pretrain(net, pretrain_file)


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
                          lr=schduler(0), momentum=0.9, weight_decay=0.0001)


    num_iters   = 300 * 1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, iter_save_interval))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        checkpoint  = torch.load(initial_optimizer)
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

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
                net.set_mode('valid')
                valid_loss = do_valid(net, valid_loader, criterion)
                net.set_mode('train')

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
                log.write('%0.4f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (\
                         rate, iter/1000, epoch,
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
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)



            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)
            net.set_mode('train')
            input = input.cuda()
            truth = truth.cuda()

            logit = data_parallel(net, input)
            loss  = criterion(logit, truth)
            precision, top = metric(logit, truth)


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
            print('%0.4f %5.1f %6.1f | %0.3f  %0.3f  %0.3f  (%0.3f)%s  | %0.3f  %0.3f  %0.3f  (%0.3f)  | %s' % (\
                         rate, iter/1000, epoch,
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



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()


    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#
