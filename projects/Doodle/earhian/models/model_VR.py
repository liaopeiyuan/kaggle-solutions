import torch
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
from models.modelZoo import *
from losses.main import *
from torch.nn.parallel.data_parallel import data_parallel

class model_Vlad(nn.Module):
    def __init__(self, num_classes=340, inchannels=3,time_length=8,model_name='resnet34_vlad'):
        super().__init__()
        if model_name == 'resnet34':
            self.basemodel = resnet34(True)
            self.basemodel.conv1 = nn.Conv2d(inchannels, 64, kernel_size=5, stride=2,padding=2)
        if model_name == 'resnet34_vlad':
            self.basemodel = resnet34(True)
            self.basemodel.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2,padding=2)

        elif model_name == 'xception':
            self.basemodel = xception(True)
        elif model_name == 'shufflenetv2':
            self.basemodel = shufflenetv2()
        elif model_name == 'mobilenet':
            self.basemodel = MobileNet()
        elif model_name == 'dpn68':
            self.basemodel = dpn68(num_classes=num_classes, pretrained=True)
        elif model_name == 'I3d':
            self.basemodel = get_I3d(True)
        elif model_name == 'seresnext50':
            self.basemodel = se_resnext50_32x4d( inchannels=inchannels,pretrained='imagenet')
        self.vald = NetVLAD(feature_size=512, max_frames=time_length,cluster_size=16)
        # self.ensemble = nn.Sequential(
        #     nn.Conv1d(512, 512, kernel_size=3,padding=1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 340, kernel_size=1),
        #
        # )
        self.fc = nn.Linear(8192, 340)
    def forward(self, x):
        b = x.size(0)
        x = x.view(x.size(0)*x.size(1), 1, x.size(2), x.size(3))
        x = self.basemodel(x)
        x = x.view(-1, 512)

        x = self.vald(x)
        x = self.fc(x)
        # x = x.view(b, -1, 512)
        # x = torch.transpose(x, 1, 2)
        # x = self.ensemble(x)
        # x = x.mean(2)
        return x

    def getLoss(self, target, result, loss_function=nn.CrossEntropyLoss()):
        self.loss = loss_function(target, result.long())

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        # raise NotImplementedError



if __name__ == '__main__':
    # cudnn.benchmark = True # This will make network slow ??
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    mobilenet = model_Vlad().cuda()
    input = torch.rand((8, 8, 128, 128)).cuda()
    out = mobilenet(input)
    print(out.shape)