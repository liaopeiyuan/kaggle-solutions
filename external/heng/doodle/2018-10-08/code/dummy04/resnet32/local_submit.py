import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0' #'3,2,1,0'

from common import *
from data   import *



##----------------------------------------
from model32_resnet34 import *




def test_augment(drawing,label,index, augment):
    cache = Struct(data = drawing.copy(), label = label, index=index)

    #<todo> ... different test-time augment ...

    image = drawing_to_image(drawing, 32, 32)
    return image, label, cache




##############################################################################################

#generate prediction npy_file
def make_npy_file_from_model(checkpoint, mode, split, augment, out_test_dir, npy_file):

    ## setup  -----------------
    # os.makedirs(out_test_dir +'/backup', exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.test.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_test_dir +'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_test_dir = %s\n' % out_test_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size     = 512 #256 #512

    test_dataset = DoodleDataset(mode, split,
                              lambda drawing, label, index : test_augment(drawing, label, index, augment),)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True,
                        collate_fn  = null_collate)

    assert(len(test_dataset)>=batch_size)
    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    if 1:
        log.write('\tcheckpoint = %s\n' % checkpoint)
        net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))


        ####### start here ##########################
        criterion = softmax_cross_entropy_criterion
        test_num  = 0
        probs    = []
        truths   = []
        losses   = []
        corrects = []

        net.set_mode('test')
        for input, truth, cache in test_loader:
            print('\r\t',test_num, end='', flush=True)
            test_num += len(truth)

            with torch.no_grad():
                input = input.cuda()
                logit = data_parallel(net,input)
                prob  = F.softmax(logit,1)
                probs.append(prob.data.cpu().numpy())


                if mode=='train': # debug only
                    truth = truth.cuda()
                    loss    = criterion(logit, truth, False)
                    correct = metric(logit, truth, False)

                    losses.append(loss.data.cpu().numpy())
                    corrects.append(correct.data.cpu().numpy())
                    truths.append(truth.data.cpu().numpy())


        assert(test_num == len(test_loader.sampler))
        print('\r\t',test_num, end='\n', flush=True)
        prob = np.concatenate(probs)

        if mode=='train': # debug only
            correct = np.concatenate(corrects)
            truth   = np.concatenate(truths).astype(np.int32).reshape(-1,1)
            loss    = np.concatenate(losses)
            loss    = loss.mean()
            correct = correct.mean(0)
            top = [correct[0], correct[0]+correct[1], correct[0]+correct[1]+correct[2]]
            precision = correct[0]/1 + correct[1]/2 + correct[2]/3
            print('top      ', top)
            print('precision', precision)
            print('')
    #-------------------------------------------


    np.save(npy_file, np_float32_to_uint8(prob))
    print(prob.shape)
    log.write('\n')






def prob_to_csv(prob, key_id, csv_file):
    top = np.argsort(-prob,1)[:,:3]
    word = []
    for (t0,t1,t2) in top:
        word.append(
            CLASS_NAME[t0] + ' ' + \
            CLASS_NAME[t1] + ' ' + \
            CLASS_NAME[t2]
        )
    df = pd.DataFrame({ 'key_id' : key_id , 'word' : word}).astype(str)
    df.to_csv(csv_file, index=False, columns=['key_id', 'word'], compression='gzip')



def npy_file_to_sbmit_csv(mode, split, npy_file, csv_file):
    print('NUM_CLASS', NUM_CLASS)
    complexity='simplified'

    if mode=='train':
        raise NotImplementedError

    if mode=='test':
        assert(NUM_CLASS==340)
        global TEST_DF

        if TEST_DF == []:
            TEST_DF = pd.read_csv(DATA_DIR + '/csv/test_%s.csv'%(complexity))
        key_id = TEST_DF['key_id'].values


    prob = np_uint8_to_float32(np.load(npy_file))
    print(prob.shape)

    prob_to_csv(prob, key_id, csv_file)
















#################################################################################################3

def run_test_fold():

    mode  = 'test' #'train'
    configures =[
         Struct(
            split        = '<NIL>', #'valid_0', #
            out_test_dir = '/root/share/project/kaggle/google_doodle/results/resnet34-fold0/test',
            checkpoint   = '/root/share/project/kaggle/google_doodle/results/resnet34-fold0/checkpoint/00004000_model.pth',
         ),
    ]


    for configure in configures:
        split        = configure.split
        out_test_dir = configure.out_test_dir
        checkpoint   = configure.checkpoint
        augment      = 'null'

        npy_file = out_test_dir + '/%s-%s.prob.uint8.npy'%(mode,augment)
        csv_file = out_test_dir + '/%s-%s.submit.csv.gz'%(mode,augment)

        make_npy_file_from_model(checkpoint, mode, split, augment, out_test_dir, npy_file)
        npy_file_to_sbmit_csv(mode, split, npy_file, csv_file)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_test_fold()


    print('\nsucess!')


