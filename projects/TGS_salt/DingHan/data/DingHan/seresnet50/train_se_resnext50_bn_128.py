import os
from common import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '3,2' #'3,2,1,0'

from data_util import *

SIZE = 101
PAD1 = 13
PAD2 = 14
Y0, Y1, X0, X1 = PAD1, PAD1 + SIZE, PAD1, PAD1 + SIZE
swa=False
##----------------------------------------
# from model_resnet34 import SaltNet as Net
# from model_resnet34_bn import SaltNet as Net
# from resnet34.model_resnet34_bn import SaltNet as Net
from seresnet50.model_se_resnext50_bn import SeResNeXt50Unet as Net

ROOT_PATH = 'E:\\DHWorkStation\\Project\\tgs_pytorch\\'
BRANCH_PATH = 'seresnext50_bn\\non_empty_test\\'


##old format
def load_old_pretrain_file(net, pretrain_file, skip=[]):
    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = net.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if any(s in key for s in skip):
            continue

        if 'resnet.' in key:
            key0 = key.replace('resnet.', 'encoder.')
            state_dict[key] = pretrain_state_dict[key0]
            continue

        found = 0
        for i in range(2, 6):
            if 'encoder%d.' % i in key:
                key0 = key.replace('encoder%d.' % i, 'conv%d.' % i)
                state_dict[key] = pretrain_state_dict[key0]
                found = 1
        if found == 1: continue

        if 'center.0.' in key:
            key0 = key.replace('center.0.', 'center.block.1.')
            if '.bn.' not in key:
                state_dict[key] = pretrain_state_dict[key0]
            continue

        if 'center.2.' in key:
            key0 = key.replace('center.2.', 'center.block.2.')
            if '.bn.' not in key:
                state_dict[key] = pretrain_state_dict[key0]
            continue

        # 'decoder5.conv1.weight'  'dec5.block.1.conv.weight'

        for i in range(5, 0, -1):
            if 'decoder%d.conv1.' % i in key:
                key0 = key.replace('decoder%d.conv1.' % i, 'dec%d.block.1.' % i)
                if '.bn.' not in key:
                    state_dict[key] = pretrain_state_dict[key0]
                found = 1
            if 'decoder%d.conv2.' % i in key:
                key0 = key.replace('decoder%d.conv2.' % i, 'dec%d.block.2.' % i)
                if '.bn.' not in key:
                    state_dict[key] = pretrain_state_dict[key0]
                found = 1
        if found == 1: continue

        if 'logit.0.' in key:
            key0 = key.replace('logit.0.', 'dec0.conv.')
            state_dict[key] = pretrain_state_dict[key0]
            continue

        if 'logit.2.' in key:
            key0 = key.replace('logit.2.', 'final.')
            state_dict[key] = pretrain_state_dict[key0]
            continue

        # print(key)
        state_dict[key] = pretrain_state_dict[key]

    net.load_state_dict(state_dict)
    return net


def valid_augment(image, mask, index):
    cache = Struct(image=image.copy(), mask=mask.copy())
    # image, mask = do_resize2(image, mask, SIZE, SIZE)
    image, mask = do_center_pad2(image, mask, PAD1, PAD2)
    return image, mask, index, cache


def train_augment(image, mask, index):
    cache = Struct(image=image.copy(), mask=mask.copy())

    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
        pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)  # 0.125

        if c == 1:
            image, mask = do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.07, 0.07))
            pass

        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))  # 10

        if c == 3:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0, 0.15))  # 0.10
            pass

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        if c == 2:
            image = do_gamma(image, np.random.uniform(1 - 0.08, 1 + 0.08))
            # if c==1:
            #     image = do_invert_intensity(image)

    # image, mask = do_resize2(image, mask, SIZE, SIZE)
    image, mask = do_center_pad2(image, mask, PAD1, PAD2)
    return image, mask, index, cache


### training ##############################################################

def do_valid(net, valid_loader):
    valid_num = 0
    valid_loss = np.zeros(3, np.float32)

    predicts = []
    truths = []

    for input, truth, index, cache in valid_loader:
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit = data_parallel(net, input)  # net(input)
            prob = F.sigmoid(logit)
            loss = net.criterion(logit, truth)
            dice = net.metric(logit, truth)

        batch_size = len(index)
        valid_loss += batch_size * np.array((loss.item(), dice.item(), 0))
        valid_num += batch_size

        prob = prob[:, :, Y0:Y1, X0:X1]
        truth = truth[:, :, Y0:Y1, X0:X1]
        prob = F.avg_pool2d(prob, kernel_size=2, stride=2)
        truth = F.avg_pool2d(truth, kernel_size=2, stride=2)
        predicts.append(prob.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())

    assert (valid_num == len(valid_loader.sampler))
    valid_loss = valid_loss / valid_num

    # --------------------------------------------------------
    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()
    precision, result, threshold = do_kaggle_metric(predicts, truths)
    valid_loss[2] = precision.mean()

    return valid_loss

def moving_average(net1, net2, alpha=1.0):
    print("averaging")
    dict=net1.state_dict()
    for k in net2.state_dict().keys():
        dict[k]=dict[k]*(1-alpha)+net2.state_dict()[k]*alpha
    net1.load_state_dict(dict)
    #     #net1.state_dict()[k]=net1.state_dict()[k]*(1-alpha)
    #     # net1.state_dict()[k]=net1.state_dict()[k]+net2.state_dict()[k]*alpha
    #     net1.state_dict()[k]=net2.state_dict()[k]
    # # for param1, param2 in zip(net1.parameters(), net2.parameters()):
    #     param1.data *= (1.0 - alpha)
    #     param1.data += param2.data * alpha
        # param1.data=param2.data
def net_equal(net1,net2):
    print("equaling")
    assert net1.state_dict().keys()==net2.state_dict().keys()
    for k in net1.state_dict().keys():
        if not torch.equal(net1.state_dict()[k],net2.state_dict()[k]):
            print('cmp!!!!')
            print(net1.state_dict()[k])
            print(net2.state_dict()[k])

def run_train():
    out_dir = ROOT_PATH + 'output\\' + BRANCH_PATH
    initial_checkpoint = \
    None
     # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\fold0\\checkpoint\\00031800_iou_0.871_model.pth'


    pretrain_file = \
    'E:\\DHWorkStation\\Project\\tgs_pytorch\\pretrained\\seresnext50\\se_resnext50_32x4d-a260b3a4.pth'
    # 'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\resnet34\\checkpoint\\00014999_model.pth'

    # None  #'/root/share/project/kaggle/tgs/results/resnet/resnet34_256/fold0/initial/00052000_model.pth'
    # '/root/share/project/kaggle/tgs/data/model/resnet34-333f7ec4.pth'

    DATA_DIR = ROOT_PATH + "data\\"
    ## setup  -----------------
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/train', exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir + '/backup/code.train.%s.zip' % IDENTIFIER)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    batch_size = 16
    train_dataset = TsgDataset('train_nonempty_2048', train_augment, 'train')
    # batch_size=int(np.round(len(train_dataset.images)/200))
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        # sampler     = ConstantSampler(train_dataset,[31]*batch_size*100),
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate)

    valid_dataset = TsgDataset('valid_nonempty_390', valid_augment, 'train')
    valid_loader = DataLoader(
        valid_dataset,
        sampler=RandomSampler(valid_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate)
    assert (len(train_dataset) >= batch_size)
    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset.split = %s\n' % (train_dataset.split))
    log.write('valid_dataset.split = %s\n' % (valid_dataset.split))
    log.write('\n')

    # debug
    if 0:  # debug  ##-------------------------------

        for input, truth, index, cache in train_loader:
            images = input.cpu().data.numpy().squeeze()
            masks = truth.cpu().data.numpy().squeeze()
            batch_size = len(index)
            for b in range(batch_size):
                image = images[b] * 255
                image = np.dstack([image, image, image])

                mask = masks[b]

                image_show('image', image, resize=2)
                image_show_norm('mask', mask, max=1, resize=2)

                overlay0 = draw_mask_overlay(mask, image, color=[0, 0, 255])
                overlay0 = draw_mask_to_contour_overlay(mask, overlay0, 2, color=[0, 0, 255])

                image_show('overlay0', overlay0, resize=2)
                cv2.waitKey(0)
    # --------------------------------------




    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()
    if swa:
        swa_net=Net().cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if swa:
            swa_net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
            swa_net.set_mode('valid')
    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        # et = load_old_pretrain_file(net, pretrain_file, skip=[])
        net.load_pretrain(pretrain_file)
    log.write('%s\n' % (type(net)))
    log.write('\n')

    ## optimiser ----------------------------------
    epochs = 360
    lr = 0.01
    # encoder_ratio=0.1
    # encoder_lr=encoder_ratio*lr

    #### unfreeze layer by layer ####
    # encoders_ratio=[0.05,0.1,0.2,0.4,0.8]
    # encoders_lr=[ratio*lr for ratio in encoders_ratio]
    #### unfreeze layer by layer ####
    iter_per_epoch=len(train_dataset.images)//batch_size
    num_iters = epochs *iter_per_epoch
    iter_smooth = 10
    iter_log = iter_per_epoch//2
    iter_valid = iter_per_epoch
    iter_save = [0, num_iters - 1] \
                +list(range(0,num_iters-1,20*iter_per_epoch))
                # + list(range(int(0.2 * epochs) * 200, int(0.5 * epochs) * 200, 50 * 200)) \
                # + list(range(int(0.5 * epochs) * 200, int(0.9 * epochs) * 200, 20 * 200)) \
                # + list(range(int(0.9 * epochs) * 200, int(0.95 * epochs) * 200,10 * 200)) \
                # + list(range(int(0.95 * epochs) * 200, num_iters, 5*200))

    # ------------------------------------------------------
    if 0:  ##freeze
        for p in net.feature_net.parameters():
            p.requires_grad = False

    # net.set_mode('train',is_freeze_bn=True)
    # ------------------------------------------------------
    # schduler=SuddenCyclicScheduler(base_lr=0.002,max_lr=0.01,step=200*25,start_step=200*50)
    schduler = StepScheduler(
        [(0, 0.01)])  # None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    # $schduler = CyclicScheduler(base_lr=0.01,max_lr=0.06,step=2000,mode='triangular2')
    # base_params = list(map(id, net.resnet.parameters()))
    # logits_params = filter(lambda p: id(p) not in base_params, net.parameters())
    # params = [
    #     {"params": logits_params, "lr": lr},
    #     {"params": net.resnet.parameters(), "lr": encoder_lr},
    # ]
    #### unfreeze layer by layer ####
    # params = [
    #     {"params": logits_params, "lr": lr},
    # ]
    # for index,layer in enumerate([net.conv1,net.encoder2,net.encoder3,net.encoder4,net.encoder5]):
    #     params.append({"params":layer,"lr":encoders_lr[index]})
    #### unfreeze layer by layer ####
    # optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=lr, momentum=0.9, weight_decay=0.0001)
    start_iter = 0
    start_epoch = 0
    swa_start_epoch=200*300
    swa_n=0
    swa_avg_epochs=200
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint.replace('_model.pth', '_optimizer.pth'))
        start_iter = checkpoint['iter']
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  # load all except learning rate
        # optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, [rate])
        pass

    ## start training here! ##############################################
    log.write('** start training here! **\n')

    # log.write(' samples_per_epoch = %d\n\n'%len(train_dataset))
    log.write(
        ' rate    iter   epoch   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write(
        '-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)
    rate = 0
    iter = 0
    i = 0
    val_loss_min = 0.300
    swa_val_loss_min = 0.250
    val_iou_max =0.800
    swa_val_iou_max = 0.875
    start = timer()
    while iter < num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6, np.float32)
        sum = 0
        optimizer.zero_grad()
        for input, truth, index, cache in train_loader:

            if 0:  # debug  ##-------------------------------

                image = input.cpu().data.numpy().squeeze()
                mask = truth.cpu().data.numpy().squeeze()

                batch_size = len(index)
                for b in range(batch_size):
                    image_show_norm('image', image[b], max=1, resize=2)
                    image_show_norm('mask', mask[b], max=1, resize=2)
                    cv2.waitKey(0)
            # --------------------------------------

            len_train_dataset = len(train_dataset)
            batch_size = len(index)
            iter = i + start_iter
            epoch = (iter - start_iter) * batch_size / len_train_dataset + start_epoch
            num_samples = epoch * len_train_dataset

            if iter % iter_valid == 0 and iter > 0:
                net.set_mode('valid')
                valid_loss = do_valid(net, valid_loader)
                if valid_loss[0] < val_loss_min:
                    val_loss_min = valid_loss[0]
                    try:
                        torch.save(net.state_dict(),
                                   out_dir + '/checkpoint/%08d_loss_%0.3f_model.pth' % (iter, valid_loss[0]))
                        torch.save({
                            'optimizer': optimizer.state_dict(),
                            'iter': iter,
                            'epoch': epoch,
                        }, out_dir + '/checkpoint/%08d_loss_%0.3f_optimizer.pth' % (iter, valid_loss[0]))
                    except:
                        pass
                elif valid_loss[2] > val_iou_max:
                    val_iou_max = valid_loss[2]
                    try:
                        torch.save(net.state_dict(),
                                   out_dir + '/checkpoint/%08d_iou_%0.3f_model.pth' % (iter, valid_loss[2]))
                        torch.save({
                            'optimizer': optimizer.state_dict(),
                            'iter': iter,
                            'epoch': epoch,
                        }, out_dir + '/checkpoint/%08d_iou_%0.3f_optimizer.pth' % (iter, valid_loss[2]))
                    except:
                        pass
                print('\r', end='', flush=True)
                net.set_mode('train')
                log.write('%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s \n' % ( \
                    # log.write('%0.9f  %5.1f  %6.1f  |  %0.9f  %0.9f  (%0.9f) |  %0.9f  %0.9f  |  %0.9f  %0.9f  | %s \n' % ( \
                    rate, iter / 1000, epoch,
                    valid_loss[0], valid_loss[1], valid_loss[2],
                    train_loss[0], train_loss[1],
                    batch_loss[0], batch_loss[1],
                    time_to_str((timer() - start))))
                time.sleep(0.01)
                if swa:
                    #print(iter, swa_start_epoch, swa_avg_epochs)
                    #print((iter) >= swa_start_epoch and ((iter - swa_start_epoch) % swa_avg_epochs) == 0)
                    if (iter) >= swa_start_epoch and ((iter - swa_start_epoch) % swa_avg_epochs) == 0 and iter!=swa_start_epoch and (valid_loss[0]<=0.25 or valid_loss[2]>=0.880):
                        moving_average(swa_net, net, 1.0 / (swa_n + 1))
                        swa_n += 1
                        #net_equal(net,swa_net)
                    if iter>=swa_start_epoch and (iter - swa_start_epoch)%swa_avg_epochs==0 and iter!=swa_start_epoch:
                        swa_net.set_mode('valid')
                        swa_valid_loss = do_valid(swa_net, valid_loader)
                        if swa_valid_loss[0] < swa_val_loss_min:
                            swa_val_loss_min = swa_valid_loss[0]
                            try:
                                torch.save(net.state_dict(),
                                           out_dir + '/checkpoint/swa_%08d_loss_%0.3f_model.pth' % (iter, swa_valid_loss[0]))
                                torch.save({
                                    'optimizer': optimizer.state_dict(),
                                    'iter': iter,
                                    'epoch': epoch,
                                }, out_dir + '/checkpoint/swa_%08d_loss_%0.3f_optimizer.pth' % (iter, swa_valid_loss[0]))
                            except:
                                pass
                        elif swa_valid_loss[2] > swa_val_iou_max:
                            swa_val_iou_max = swa_valid_loss[2]
                            try:
                                torch.save(net.state_dict(),
                                           out_dir + '/checkpoint/swa_%08d_iou_%0.3f_model.pth' % (iter, swa_valid_loss[2]))
                                torch.save({
                                    'optimizer': optimizer.state_dict(),
                                    'iter': iter,
                                    'epoch': epoch,
                                }, out_dir + '/checkpoint/swa_%08d_iou_%0.3f_optimizer.pth' % (iter, swa_valid_loss[2]))
                            except:
                                pass
                        print('\r', end='', flush=True)
                        swa_net.set_mode('train')
                        log.write(
                            'swa  %0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s \n' % ( \
                                # log.write('%0.9f  %5.1f  %6.1f  |  %0.9f  %0.9f  (%0.9f) |  %0.9f  %0.9f  |  %0.9f  %0.9f  | %s \n' % ( \
                                rate, iter / 1000, epoch,
                                swa_valid_loss[0], swa_valid_loss[1], swa_valid_loss[2],
                                train_loss[0], train_loss[1],
                                batch_loss[0], batch_loss[1],
                                time_to_str((timer() - start))))
                        time.sleep(0.01)


            # if 1:
            if iter in iter_save:
                try:
                    torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (iter))
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter': iter,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_optimizer.pth' % (iter))
                except:
                    pass
                pass
                if swa and iter>=swa_start_epoch:
                    try:
                        torch.save(net.state_dict(), out_dir + '/checkpoint/swa_%08d_model.pth' % (iter))
                        torch.save({
                            'optimizer': optimizer.state_dict(),
                            'iter': iter,
                            'epoch': epoch,
                        }, out_dir + '/checkpoint/swa_%08d_optimizer.pth' % (iter))
                    except:
                        pass
                    pass

            # learning rate schduler -------------
            if schduler is not None:
                lr = schduler.get_rate(iter)
                # if lr >=0.01:
                #     encoder_lr=encoder_ratio*lr
                # else:
                #     encoder_lr=lr
                if lr < 0: break
                adjust_learning_rate(optimizer, [lr])
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            # net.set_mode('train',is_freeze_bn=True)
            net.set_mode('train')

            input = input.cuda()
            truth = truth.cuda()
            logit = data_parallel(net, input)  # net(input)
            loss = net.criterion(logit, truth)
            dice = net.metric(logit, truth)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # torch.nn.utils.clip_grad_norm(net.parameters(), 1)


            # print statistics  ------------
            batch_loss = np.array((
                loss.item(),
                dice.item(),
                0, 0, 0, 0,
            ))
            sum_train_loss += batch_loss
            sum += 1
            if iter % iter_smooth == 0:
                train_loss = sum_train_loss / sum
                sum_train_loss = np.zeros(6, np.float32)
                sum = 0

            print('\r%0.4f  %5.1f  %6.1f  |  %0.3f  %0.3f  (%0.3f) |  %0.3f  %0.3f  |  %0.3f  %0.3f  | %s ' % ( \
                rate, iter / 1000, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2],
                train_loss[0], train_loss[1],
                batch_loss[0], batch_loss[1],
                time_to_str((timer() - start))), end='', flush=True)
            i = i + 1

            # <debug> ===================================================================
            if 0:
                # if iter%200==0:
                # voxel, aux, query, link, truth, cache = make_valid_batch(valid_dataset.dataset, batch_size=2)

                net.set_mode('test')  #
                with torch.no_grad():
                    logit = net(input)
                    prob = F.sigmoid(logit)
                    loss = net.criterion(logit, truth)
                    dice = net.metric(logit, truth)

                    if 0:
                        loss = net.criterion(logit, truth)
                        accuracy, hit_rate, precision_rate = net.metric(logit, truth)
                        valid_loss[0] = loss.item()
                        valid_loss[1] = accuracy.item()
                        valid_loss[2] = hit_rate.item()
                        valid_loss[3] = precision_rate.item()

                        # show only b in batch ---
                b = 1
                prob = prob.data.cpu().numpy()[b].squeeze()
                truth = truth.data.cpu().numpy()[b].squeeze()
                input = input.data.cpu().numpy()[b].squeeze()

                all = np.hstack([input, truth, prob])
                image_show_norm('all', all, max=1, resize=3)
                cv2.waitKey(100)

                net.set_mode('train')
                # <debug> ===================================================================

        pass  # -- end of one data loader --
    pass  # -- end of all iterations --

    if 1:  # save last
        torch.save(net.state_dict(), out_dir + '/checkpoint/%d_model.pth' % (i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter': i,
            'epoch': epoch,
        }, out_dir + '/checkpoint/%d_optimizer.pth' % (i))
        if swa:
            torch.save(net.state_dict(), out_dir + '/checkpoint/swa_%d_model.pth' % (i))
            torch.save({
                'optimizer': optimizer.state_dict(),
                'iter': i,
                'epoch': epoch,
            }, out_dir + '/checkpoint/swa_%d_optimizer.pth' % (i))

    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')



# ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#
