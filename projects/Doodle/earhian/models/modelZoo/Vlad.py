import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

class NetVLAD(nn.Module):
    def __init__(self, feature_size=512, max_frames=8,cluster_size=16, add_bn=False, truncate=False):
        super(NetVLAD, self).__init__()
        self.feature_size = int(feature_size // 2) if truncate else feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = nn.BatchNorm1d(cluster_size, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(self.feature_size, self.cluster_size)
        self.softmax = nn.Softmax(dim=1)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.feature_size,
                                                               self.cluster_size)).float()
        self.add_bn = add_bn
        self.truncate = truncate
        self.first = True
        self.init_parameters()

    def init_parameters(self):
        init.normal_(self.cluster_weights2, std=1 / math.sqrt(self.feature_size))

    def forward(self, reshaped_input):
        random_idx = torch.bernoulli(torch.Tensor([0.5]))
        if self.truncate:
            if self.training == True:
                reshaped_input = reshaped_input[:, :self.feature_size].contiguous() if random_idx[0]==0 else reshaped_input[:, self.feature_size:].contiguous()
            else:
                if self.first == True:
                    reshaped_input = reshaped_input[:, :self.feature_size].contiguous()
                else:
                    reshaped_input = reshaped_input[:, self.feature_size:].contiguous()
        activation = self.linear(reshaped_input)
        if self.add_bn:
            activation = self.batch_norm(activation)
        activation = self.softmax(activation).view([-1, self.max_frames, self.cluster_size])
        a_sum = activation.sum(-2).unsqueeze(1)
        a = torch.mul(a_sum, self.cluster_weights2)
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = reshaped_input.view([-1, self.max_frames, self.feature_size])
        vlad = torch.matmul(activation, reshaped_input).permute(0, 2, 1).contiguous()
        vlad = vlad.sub(a).view([-1, self.cluster_size * self.feature_size])
        if self.training == False:
            self.first = 1 - self.first
        return vlad
if __name__ == '__main__':
    # cudnn.benchmark = True # This will make network slow ??

    mobilenet = NetVLAD().cuda()
    input = torch.rand((64,512)).cuda()
    out = mobilenet(input)
    print(out.shape)