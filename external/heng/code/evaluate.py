from common import *

### metric #################################################################################
# https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/metrics.py
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/61550
# https://www.kaggle.com/c/tgs-salt-identification-challenge#evaluation


#rle encode/edcode
def do_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b>prev+1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    #if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
    #    rle[-2] = rle[-2] -1

    rle = ' '.join([str(r) for r in rle])
    return rle

def do_length_decode(rle, H, W, fill_value=255):

    mask = np.zeros((H,W), np.uint8)
    if rle=='': return mask

    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask



def do_kaggle_metric(predict,truth, threshold=0.5):

    N = len(predict)
    predict = predict.reshape(N,-1)
    truth   = truth.reshape(N,-1)

    predict = predict>threshold
    truth   = truth>0.5
    intersection = truth & predict
    union        = truth | predict
    iou = intersection.sum(1)/(union.sum(1)+EPS)

    #-------------------------------------------
    result = []
    precision = []
    is_empty_truth   = (truth.sum(1)==0)
    is_empty_predict = (predict.sum(1)==0)

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou>=t

        tp  = (~is_empty_truth)  & (~is_empty_predict) & (iou> t)
        fp  = (~is_empty_truth)  & (~is_empty_predict) & (iou<=t)
        fn  = (~is_empty_truth)  & ( is_empty_predict)
        fp_empty = ( is_empty_truth)  & (~is_empty_predict)
        tn_empty = ( is_empty_truth)  & ( is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append( np.column_stack((tp,fp,fn,tn_empty,fp_empty)) )
        precision.append(p)

    result = np.array(result).transpose(1,2,0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold



### run ############################################################################

def run_check_run_length_encode1():
    mask =  np.ones((96,96),np.bool)
    encoding = run_length_encode(mask>0)
    print(encoding)



def run_check_run_length_encode():

    #reference encoding
    name = ['575d24d81d','a266a2a9df','75efad62c1','34e51dba6a']
    df = pd.read_csv ( '/root/share/project/kaggle/tgs/data/train.csv')
    #df = df.loc[df['id'].isin(name)]
    df = df.fillna('')


    num_false = 0
    num       = 0
    for n in df.id.values:
        mask_file = '/root/share/project/kaggle/tgs/data/train/masks/' + n +'.png'
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)


        ##--------------------------------------------------------
        if 0:
            multi_mask = skimage.morphology.label(mask>0)
            encoding = []
            num_mask = int(multi_mask.max())+1
            for m in range(1,num_mask):
                rle = run_length_encode(multi_mask==m)
                encoding.append(rle)
            if encoding==[]: encoding=['']
            encoding.sort()

        if 1:
            encoding = run_length_encode(mask>0)

        ##--------------------------------------------------------


        reference = df.loc[df['id']==n].rle_mask.values[0]
        #reference.sort()

        num_false += reference!=reference
        num += 1

        print(encoding)
        print(reference)
        print(encoding==reference)
        print('')
    ##------------------
    print('num_false ', num_false)
    print('num       ', num)



def run_check_run_length_decode():


    #reference encoding
    #name = ['575d24d81d','a266a2a9df','75efad62c1','34e51dba6a']
    #name = ['a266a2a9df','75efad62c1','34e51dba6a']
    df = pd.read_csv ( '/root/share/project/kaggle/tgs/data/train.csv')
    #df = df.loc[df['id'].isin(name)]
    df = df.fillna('')


    num_false = 0
    num       = 0
    for n in df.id.values:
        mask_file = '/root/share/project/kaggle/tgs/data/train/masks/' + n +'.png'
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        #mask = mask>0

        rle = df.loc[df['id']==n].rle_mask.values[0]
        reference = run_length_decode(rle, H=101, W=101, fill_value=255)
        #reference = reference>0

        print(np.any(reference!=mask))

        num_false += np.any(reference!=mask)
        num += 1


        # image_show('reference', reference,2)
        # image_show('mask', mask,2)
        # image_show('diff', (reference!=mask)*255,2)
        # cv2.waitKey(0)

    ##------------------
    print('')
    print('num_false ', num_false)
    print('num       ', num)









# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    #run_check_run_length_encode()
    #run_check_run_length_decode()

 
