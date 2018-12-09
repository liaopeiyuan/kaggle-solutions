import torch
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
from models.modelZoo import *
from losses.main import *
from torch.nn.parallel.data_parallel import data_parallel


def get_weaky_label(label, num_classes=340):
    b = len(label)
    zeros = torch.zeros((b, num_classes)).cuda().float()
    for i in range(b):
        zeros[i, label[i]] = 1
    return zeros
class model_QDDR(nn.Module):
    def __init__(self, num_classes=340, inchannels=1,model_name='resnet34'):
        super().__init__()
        if model_name == 'resnet34':
            self.basemodel = tvm.resnet34(True)
            if inchannels == 1:
                self.basemodel.conv1 = nn.Conv2d(inchannels, 64, kernel_size=5, stride=2,padding=2)
            self.basemodel.avgpool = nn.AdaptiveAvgPool2d(1)
            self.basemodel.fc = nn.Sequential(
                                nn.Dropout(0.2),
                                nn.Linear(512, num_classes)
                                              )
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

    def forward(self, x):
        x = data_parallel(self.basemodel, x)
        return x

    def getLoss(self, target, result, loss_function=nn.CrossEntropyLoss()):
        # self.loss = nn.CrossEntropyLoss()(target, result.long())
        self.loss = loss_function(target, result.long())

    def mean_teacher_loss(self, outs_student, outs_teacher, recognized, labels, con_weight=1):
        b = len(recognized)
        isrecognized_index = torch.nonzero(recognized).view(-1)
        nonrecognized_index = torch.nonzero(recognized == 0).view(-1)
        self.loss = (len(isrecognized_index)/b) * nn.CrossEntropyLoss()(outs_student[isrecognized_index], labels[isrecognized_index].long())
        len_nonrecognized = len(nonrecognized_index)
        if len_nonrecognized > 0:
            outs_teacher = Variable(outs_teacher.data, requires_grad=False)
            labels_teacher = labels[nonrecognized_index]
            outs_s = outs_student[nonrecognized_index]
            weaky_label = get_weaky_label(labels_teacher)
            target = (torch.softmax(outs_teacher, 1) + weaky_label)/2
            target = torch.clamp(target, 0.0, 1.0)
            self.loss += (len(nonrecognized_index)/b) *F.binary_cross_entropy(torch.softmax(outs_s, 1), target) * con_weight



    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        # raise NotImplementedError


