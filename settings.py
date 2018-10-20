from datetime import datetime

PATH= "kail"
"""
Local
"""

if PATH=='kail':
    print("Using paths on kail-main")
    CHECKPOINTS='/data/kaggle/salt/checkpoints'
    DATA='/data/kaggle/salt/'
    RESULT='/data/ml-arsenal/projects/TGS_salt'
    CODE='/data/ml-arsenal'
    CUDA_DEVICES='0,1'
    MODE='gpu'
    GRAPHICS=True
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='kail0':
    print("Using paths on kail-main w. GTX 1080 Ti")
    CHECKPOINTS='/data/kaggle/salt/checkpoints'
    DATA='/data/kaggle/salt/'
    RESULT='/data/ml-arsenal/projects/TGS_salt'
    CODE='/data/ml-arsenal'
    CUDA_DEVICES='0'
    MODE='gpu'
    GRAPHICS=True
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='kail1':
    print("Using paths on kail-main w. GTX 1070")
    CHECKPOINTS='/data/kaggle/salt/checkpoints'
    DATA='/data/kaggle/salt/'
    RESULT='/data/ml-arsenal/projects/TGS_salt'
    CODE='/data/ml-arsenal'
    CUDA_DEVICES='1'
    MODE='gpu'
    GRAPHICS=True
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='local':
    print("Using local paths on alexanderliao@alexanderliao-Thinkpad-P50.")
    CHECKPOINTS='/home/alexanderliao/data/Kaggle/competitions/tgs-salt-identification-challenge/checkpoints'
    DATA='/home/alexanderliao/data/Kaggle/competitions/tgs-salt-identification-challenge'
    RESULT='/home/alexanderliao/data/GitHub/ml-arsenal/projects/TGS_salt'
    CODE='/home/alexanderliao/data/GitHub/ml-arsenal'
    CUDA_DEVICES='0'
    MODE='gpu'
    GRAPHICS=True
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='gcp0':
    print("Using GCP paths on liaop20@kaggle.")
    CHECKPOINTS='/home/liaop20/data/salt/checkpoints'
    DATA='/home/liaop20/data/salt'
    RESULT='/home/liaop20/ml-arsenal/projects/TGS_salt'
    CODE='/home/liaop20/ml-arsenal'
    CUDA_DEVICES='0'
    MODE='gpu'
    GRAPHICS=False
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='gcp1':
    print("Using GCP paths on liaop20@kaggle.")
    CHECKPOINTS='/home/liaop20/data/salt/checkpoints'
    DATA='/home/liaop20/data/salt'
    RESULT='/home/liaop20/ml-arsenal/projects/TGS_salt'
    CODE='/home/liaop20/ml-arsenal'
    CUDA_DEVICES='1'
    MODE='gpu'
    GRAPHICS=False
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='gcp2':
    print("Using GCP paths on liaop20@kaggle.")
    CHECKPOINTS='/home/liaop20/data/salt/checkpoints'
    DATA='/home/liaop20/data/salt'
    RESULT='/home/liaop20/ml-arsenal/projects/TGS_salt'
    CODE='/home/liaop20/ml-arsenal'
    CUDA_DEVICES='2'
    MODE='gpu'
    GRAPHICS=False
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='gcp3':
    print("Using GCP paths on liaop20@kaggle.")
    CHECKPOINTS='/home/liaop20/data/salt/checkpoints'
    DATA='/home/liaop20/data/salt'
    RESULT='/home/liaop20/ml-arsenal/projects/TGS_salt'
    CODE='/home/liaop20/ml-arsenal'
    CUDA_DEVICES='3'
    MODE='gpu'
    GRAPHICS=False
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='gcp':
    print("Using GCP paths on liaop20@kaggle.")
    CHECKPOINTS='/home/liaop20/data/salt/checkpoints'
    DATA='/home/liaop20/data/salt'
    RESULT='/home/liaop20/ml-arsenal/projects/TGS_salt'
    CODE='/home/liaop20/ml-arsenal'
    CUDA_DEVICES='0,1,2,3'
    MODE='gpu'
    GRAPHICS=False
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if PATH=='aaron':
    print("Using paths on Aaron's PC.")
    CHECKPOINTS='/mydisk/Programming/Git/salt/checkpoints'
    DATA='/mydisk/Programming/Git/salt'
    RESULT='/mydisk/Programming/Git/ml-arsenal/projects/TGS_salt'
    CODE='/mydisk/Programming/Git/ml-arsenal'
    CUDA_DEVICES='0'
    MODE='gpu'
    GRAPHICS=False
    IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


print('')
