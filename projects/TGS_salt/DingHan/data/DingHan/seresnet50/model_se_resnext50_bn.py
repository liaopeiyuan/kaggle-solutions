from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d
from torch.utils import model_zoo

from common import *
from net import lovasz_losses as L

#  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#  resnet18 :  BasicBlock, [2, 2, 2, 2]
#  resnet34 :  BasicBlock, [3, 4, 6, 3]
#  resnet50 :  Bottleneck  [3, 4, 6, 3]
#
# https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
# https://github.com/ternaus/TernausNetV2
# https://github.com/neptune-ml/open-solution-salt-detection
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution
##############################################################3
#  https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py
#  https://pytorch.org/docs/stable/torchvision/models.html


"""
Code imported from https://github.com/Cadene/pretrained-models.pytorch
"""

__all__ = ['SEResNet50', 'SEResNet101', 'SEResNeXt50', 'SEResNeXt101']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64.)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = SynchronizedBatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context
class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale)
class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output

class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output

class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.BatchNorm2d(out_features),
                                     nn.ReLU(),
                                     BaseOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                                           key_channels=out_features // 2, value_channels=out_features,
                                                           dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(out_features),
                                   nn.ReLU(),)
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),)
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),)
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),)

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ,OC=False,dilation=(4,8,12)):
        super(Decoder, self).__init__()
        self.oc=OC
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        # self.conv1 = ConvBn2dV2(in_channels, channels, kernel_size=3, padding=1)
        # self.conv2 = ConvBn2dV2(channels, out_channels, kernel_size=3, padding=1)
        self.context = nn.Sequential(
                ASP_OC_Module(out_channels,out_channels,dilations=dilation)
                )
        # self.context = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     BaseOC_Module(in_channels=out_channels, out_channels=out_channels, key_channels=out_channels//2, value_channels=out_channels-out_channels//2,
        #                   dropout=0.05, sizes=([1]))
        # )
        self.scse_gate = SCSEBlock(out_channels)


    def forward(self, x):

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        if self.oc:
            y = self.context(x)
        # x= self.conv1(F.elu(x,inplace=True))
        # x = self.conv2(F.elu(x, inplace=True))
        x = self.scse_gate(x)
        if self.oc:
            return torch.cat((x,y),1)
        else:
            return x


class SeResNeXt50Unet(nn.Module):
    def load_pretrain(self, pretrain_file):
        pretrain_dict = torch.load(pretrain_file)
        state_dict = {}
        keys = list(pretrain_dict.keys())
        for key in keys:
            if 'last_layer' in key:
                continue
            state_dict[(key)] = pretrain_dict[key]
        self.encoder.load_state_dict(state_dict)


    def __init__(self,dilation=False):
        super().__init__()
        self.dilation = dilation
        self.encoder = se_resnext50_32x4d(1000)
        self.encoder1 = self.encoder.layer0
        self.encoder2 = self.encoder.layer1  # 256
        self.encoder3 = self.encoder.layer2  # 512
        self.encoder4 = self.encoder.layer3  # 1024
        self.encoder5 = self.encoder.layer4  # 2048
        self.center = nn.Sequential(
            ConvBn2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if self.dilation:
            self.center1 = nn.Sequential(
                ConvBn2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True)
            )
            self.center2 = nn.Sequential(
                ConvBn2d(512, 512, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(inplace=True)
            )

        # self.center5 = nn.Sequential(
        #     ConvBn2d(128, 128, kernel_size=3, padding=1, dilation=32),
        #     nn.ReLU(inplace=True)
        # )
        self.decoder5 = Decoder(2048 + 512, 512, 64,OC=True,dilation=(2,4,6))
        self.decoder4 = Decoder(128 + 1024, 256, 64,OC=True,dilation=(4,8,12))
        self.decoder3 = Decoder(128 + 512, 128, 64)
        self.decoder2 = Decoder(64 + 256, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(64*7, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x - mean[0]) / std[0],
            (x - mean[1]) / std[1],
            (x - mean[2]) / std[2],
        ], 1)
        # print('x',x.size())
        e1 = self.encoder1(x)  # ; print('e1',e1.size())
        e2 = self.encoder2(e1)  # ; print('e2',e2.size())
        e3 = self.encoder3(e2)  # ; print('e3',e3.size())
        e4 = self.encoder4(e3)  # ; print('e4',e4.size())
        e5 = self.encoder5(e4)  # ; print('e5',e5.size())
        f = self.center(e5)  #; print('center',f.size())
        if self.dilation:
            f1= self.center1(f)#; print('center',f1.size())
            f2=self.center2(f1)#; print('center',f2.size())
            # f3=self.center3(f2); print('center',f3.size())
            # f4=self.center4(f3); print('center',f4.size())
            #f5=self.center5(f4)
            f = torch.add(f,1,f1)
            f = torch.add(f,1,f2)
        # f=torch.cat((
        #     f,
        #     f1,
            # f2,
            # f3,
            # f4,
        # ),1)
        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        d5 = self.decoder5(torch.cat([f, e5], 1))  # ; print('d5',d5.size())
        d5 = F.upsample(d5, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.decoder4(torch.cat([d5, e4], 1))  # ; print('d4',d4.size())
        d4 = F.upsample(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d4, e3], 1))  # ; print('d3',d3.size())
        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.decoder2(torch.cat([d3, e2], 1))  # ; print('d2',d2.size())
        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(d2)  # ; print('d1',d1.size())

        f = torch.cat((
            d1,
            F.upsample(d2, scale_factor=1, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=8, mode='bilinear', align_corners=False)
        ), 1)
        f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)  # ; print('logit',logit.size())
        return logit

    def criterion(self, logit, truth):

        # loss = PseudoBCELoss2d()(logit, truth)
        # loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
        # return loss
        loss = L.lovasz_hinge(logit, truth)
        return loss

    # def criterion(self,logit, truth):
    #
    #     loss = F.binary_cross_entropy_with_logits(logit, truth)
    #     return loss



    def metric(self, logit, truth, threshold=0.5):
        prob = F.sigmoid(logit)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


class SENet(nn.Module):
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=None):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        # layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
        #                                             ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        if num_classes:
            self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

        # def features(self, x):
        #     x = self.layer0(x)
        #     x = self.layer1(x)
        #     x = self.layer2(x)
        #     x = self.layer3(x)
        #     x = self.layer4(x)
        #     return x
        #
        # def logits(self, x):
        #     x = self.avg_pool(x)
        #     if self.dropout is not None:
        #         x = self.dropout(x)
        #     x = x.view(x.size(0), -1)
        #     x = self.last_linear(x)
        #     return x
        #
        # def forward(self, x):
        #     x = self.features(x)
        #     x = self.logits(x)
        #     return x


def load_pretrained_model(model, file_path):
    model.load_state_dict(file_path)


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    # if pretrained is not None:
    #     # settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
    #     # initialize_pretrained_model(model, num_classes, settings)
    #     pretrained_state=torch.load(pretrained)
    #     model.load_state_dict(pretrained_state)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


##################### Model Definition #########################


class SEResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(SEResNet50, self).__init__()
        self.loss = loss
        base = se_resnet50()
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class SEResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(SEResNet101, self).__init__()
        self.loss = loss
        base = se_resnet101()
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class SEResNeXt50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(SEResNeXt50, self).__init__()
        self.loss = loss
        self.senet_base = se_resnext50_32x4d()
        self.base = nn.Sequential(*list(self.senet_base.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class SEResNeXt101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(SEResNeXt101, self).__init__()
        self.loss = loss
        base = se_resnext101_32x4d()
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def run_check_net():
    batch_size = 8
    C, H, W = 1, 128, 128

    input = np.random.uniform(0, 1, (batch_size, C, H, W)).astype(np.float32)
    truth = np.random.choice(2, (batch_size, C, H, W)).astype(np.float32)

    # ------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()

    # ---
    net = SeResNeXt50Unet().cuda()
    net.set_mode('train')
    # print(net)
    # exit(0)

    net.load_pretrain(
        'E:\\DHWorkStation\\Project\\tgs_pytorch\\pretrained\\seresnext50\\se_resnext50_32x4d-a260b3a4.pth')

    logit = net(input)
    loss = net.criterion(logit, truth)
    dice = net.metric(logit, truth)

    print('loss : %0.8f' % loss.item())
    print('dice : %0.8f' % dice.item())
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0001)

    # optimizer = optim.Adam(net.parameters(), lr=0.001)


    i = 0
    optimizer.zero_grad()
    while i <= 500:

        logit = net(input)
        loss = net.criterion(logit, truth)
        dice = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 20 == 0:
            print('[%05d] loss, dice  :  %0.5f,%0.5f' % (i, loss.item(), dice.item()))
        i = i + 1

#
# run_check_net()
# ### without pretrain
# [00000]
# loss, dice: 1.01400, 0.50066
# [00020]
# loss, dice: 1.00079, 0.49958
# [00040]
# loss, dice: 1.00004, 0.49958
# [00060]
# loss, dice: 0.99982, 0.49958
# [00080]
# loss, dice: 0.99952, 0.49958
# [00100]
# loss, dice: 0.99900, 0.49958
# [00120]
# loss, dice: 0.99783, 0.49958
# [00140]
# loss, dice: 0.99408, 0.49958
# [00160]
# loss, dice: 0.97106, 0.49958
# [00180]
# loss, dice: 0.87424, 0.55930
# [00200]
# loss, dice: 0.78611, 0.53180
# [00220]
# loss, dice: 0.71940, 0.78762
# [00240]
# loss, dice: 0.66982, 0.83044
# [00260]
# loss, dice: 0.67656, 0.82647
# [00280]
# loss, dice: 0.59726, 0.85464
# [00300]
# loss, dice: 0.36389, 0.91353
# [00320]
# loss, dice: 0.25008, 0.94157
# [00340]
# loss, dice: 0.21201, 0.94981
# [00360]
# loss, dice: 0.19738, 0.95384
# [00380]
# loss, dice: 0.17988, 0.95844
# [00400]
# loss, dice: 0.15965, 0.96169
# [00420]
# loss, dice: 0.15145, 0.96404
# [00440]
# loss, dice: 0.14283, 0.96663
# [00460]
# loss, dice: 0.13184, 0.96834
# [00480]
# loss, dice: 0.12259, 0.97068
# [00500]
# loss, dice: 0.11979, 0.97170
