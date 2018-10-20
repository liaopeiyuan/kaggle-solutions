from include import *
from utility.draw import *
from utility.file import *

from net.rate import *
from net.layer.function import *
from net.loss   import *
from net.metric import *


from dataset.transform import *
from dataset.sampler import *



# kaggle ---
from evaluate import *
from utility.draw import *
#from draw import *





#---------------------------------------------------------------------------------

class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))

if 1:
    SEED = 35202  #123  #35202   #int(time.time()) #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)

if 1:

    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    #print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


print('')

#---------------------------------------------------------------------------------
