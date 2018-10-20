from dependencies import random, np, torch, os
from settings import *

print("Fixing random seed for reproducibility...")
SEED = 35202  #123  #35202   #int(time.time()) #
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print ('\tSetting random seed to {}.'.format(SEED))
print('')

torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.enabled   = True
print ('Setting CUDA environment...')
print ('\ttorch.__version__              =', torch.__version__)
print ('\ttorch.version.cuda             =', torch.version.cuda)
print ('\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())

os.environ['CUDA_VISIBLE_DEVICES']=CUDA_DEVICES
if MODE=='cpu':
    print("Warning: you've set the mode to CPU; \nthe code won't run on NVIDIA GPU even the CUDA and CUDNN queries are successful.")
try:
    print ('\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
    NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

except Exception:
    print ('\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
    NUM_CUDA_DEVICES = 1

print ('\ttorch.cuda.device_count()      =', torch.cuda.device_count())


print('')