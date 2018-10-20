import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable

class VGG16Modified(nn.Module):

    def __init__(self, n_classes=21):
        super(VGG16Modified, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2 128

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4 64

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8 32

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16 16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32 8


        self.conv6s_re = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.conv6_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv6_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv6_1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))

        self.conv7_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv7_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv7_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))

        self.conv8_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv8_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv8_1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))

        self.conv9 = nn.Sequential(nn.Conv2d(128, 32*3*4, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.Tanh())

        self.conv10 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(nn.Conv2d(64, n_classes, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))



        self.coarse_conv_in = nn.Sequential(nn.Conv2d(n_classes, 32, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(kernel_size=2, stride=2))
        self.copy_params_from_vgg16()

    def to_tridiagonal_multidim(self, w):
        N,W,C,D = w.size()
        tmp_w = w / torch.sum(torch.abs(w),dim=3).unsqueeze(-1)
        tmp_w = tmp_w.unsqueeze(2).expand([N,W,W,C,D])

        eye_a = Variable(torch.diag(torch.ones(W-1).cuda(),diagonal=-1))
        eye_b = Variable(torch.diag(torch.ones(W).cuda(),diagonal=0))
        eye_c = Variable(torch.diag(torch.ones(W-1).cuda(),diagonal=1))

        
        tmp_eye_a = eye_a.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
        a = tmp_w[:,:,:,:,0] * tmp_eye_a
        tmp_eye_b = eye_b.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
        b = tmp_w[:,:,:,:,1] * tmp_eye_b
        tmp_eye_c = eye_c.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
        c = tmp_w[:,:,:,:,2] * tmp_eye_c

        return a+b+c
    def forward(self, x, coarse_segmentation):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        conv3_1 = self.relu3_1(self.conv3_1(h))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3= self.relu3_3(self.conv3_3(conv3_2))
        h = self.pool3(conv3_3)
        pool3 = h  # 1/8

        conv4_1 = self.relu4_1(self.conv4_1(h))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        h = self.pool4(conv4_3)
        pool4 = h  # 1/16

        conv5_1 = self.relu5_1(self.conv5_1(h))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))
        h = self.pool5(conv5_3)



        conv6_re = self.conv6s_re(h)



        skip_1 = conv5_3 + conv6_re
        conv6_3 = self.conv6_3(skip_1)        
        skip_2 = conv5_2 + conv6_3
        conv6_2 = self.conv6_2(skip_2)        
        skip_3 = conv5_1 + conv6_2
        conv6_1 = self.conv6_1(skip_3)

        skip_4 = conv4_3 + conv6_1
        conv7_3 = self.conv7_3(skip_4)        
        skip_5 = conv4_2 + conv7_3
        conv7_2 = self.conv7_2(skip_5)        
        skip_6 = conv4_1 + conv7_2
        conv7_1 = self.conv7_1(skip_6)

        skip_7 = conv3_3 + conv7_1
        conv8_3 = self.conv8_3(skip_7)        
        skip_8 = conv3_2 + conv8_3
        conv8_2 = self.conv8_2(skip_8)        
        skip_9 = conv3_1 + conv8_2
        conv8_1 = self.conv8_1(skip_9)

        conv9 = self.conv9(conv8_1)

        N,C,H,W = conv9.size()
        four_directions = C // 4
        conv9_reshaped_W = conv9.permute(0,2,3,1)
        # conv9_reshaped_H = conv9.permute(0,3,2,1)

        conv_x1_flat = conv9_reshaped_W[:,:,:,0:four_directions].contiguous()
        conv_y1_flat = conv9_reshaped_W[:,:,:,four_directions:2*four_directions].contiguous()
        conv_x2_flat = conv9_reshaped_W[:,:,:,2*four_directions:3*four_directions].contiguous()
        conv_y2_flat = conv9_reshaped_W[:,:,:,3*four_directions:4*four_directions].contiguous()

        w_x1 = conv_x1_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3
        w_y1 = conv_y1_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3
        w_x2 = conv_x2_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3
        w_y2 = conv_y2_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3

        rnn_h1 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h2 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h3 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h4 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())

        x_t = self.coarse_conv_in(coarse_segmentation).permute(0,2,3,1)
        
        
        # horizontal
        for i in range(W):
            #left to right
            tmp_w = w_x1[:,:,i,:,:]  # N, H, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, H, W, 32 
            # tmp_x = x_t[:,:,i,:].unsqueeze(1)
            # tmp_x = tmp_x.expand([batch, W, H, 32])
            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h1[:,:,i-1,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)


            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t[:,:,i,:]

            rnn_h1[:,:,i,:] = w_x_curr + w_h_prev


            #right to left
            # tmp_w = w_x1[:,:,i,:,:]  # N, H, 1, 32, 3 
            # tmp_w = to_tridiagonal_multidim(tmp_w)

            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h2[:,:,W - i,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)
   

            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t[:,:,W - i-1,:]
            rnn_h2[:,:,W - i-1,:] = w_x_curr + w_h_prev 

        w_y1_T = w_y1.transpose(1,2)
        x_t_T = x_t.transpose(1,2)

        for i in range(H):
            # up to down
            tmp_w = w_y1_T[:,:,i,:,:]  # N, W, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, W, H, 32 

            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h3[:,:,i-1,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t_T[:,:,i,:]
            rnn_h3[:,:,i,:] = w_x_curr + w_h_prev

            # down to up
            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h4[:,:,H - i,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t[:,:,H-i-1,:]
            rnn_h4[:,:,H-i-1,:] = w_x_curr + w_h_prev  

        rnn_h3 = rnn_h3.transpose(1,2)
        rnn_h4 = rnn_h4.transpose(1,2)

        rnn_h5 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h6 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h7 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h8 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())

        # horizontal
        for i in range(W):
            #left to right
            tmp_w = w_x2[:,:,i,:,:]  # N, H, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, H, W, 32 
            # tmp_x = x_t[:,:,i,:].unsqueeze(1)
            # tmp_x = tmp_x.expand([batch, W, H, 32])
            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h5[:,:,i-1,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h1[:,:,i,:]
            rnn_h5[:,:,i,:] = w_x_curr + w_h_prev


            #right to left
            # tmp_w = w_x1[:,:,i,:,:]  # N, H, 1, 32, 3 
            # tmp_w = to_tridiagonal_multidim(tmp_w)
            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h6[:,:,W-i,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h2[:,:,W - i-1,:]
            rnn_h6[:,:,W - i-1,:] = w_x_curr + w_h_prev  

        w_y2_T = w_y2.transpose(1,2)
        rnn_h3_T = rnn_h3.transpose(1,2)
        rnn_h4_T = rnn_h4.transpose(1,2)
        for i in range(H):
            # up to down
            tmp_w = w_y2_T[:,:,i,:,:]  # N, W, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, W, H, 32 
            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h7[:,:,i-1,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h3_T[:,:,i,:]
            rnn_h7[:,:,i,:] = w_x_curr + w_h_prev

            # down to up
            if i == 0 :
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h8[:,:,H-i,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h4_T[:,:,H-i-1,:]
            rnn_h8[:,:,H-i-1,:] = w_x_curr + w_h_prev   

        rnn_h3 = rnn_h3.transpose(1,2)
        rnn_h4 = rnn_h4.transpose(1,2)

        concat6 = torch.cat([rnn_h5.unsqueeze(4),rnn_h6.unsqueeze(4),rnn_h7.unsqueeze(4),rnn_h8.unsqueeze(4)],dim=4)
        elt_max = torch.max(concat6, dim=4)[0]
        elt_max_reordered = elt_max.permute(0,3,1,2)
        conv10 = self.conv10(elt_max_reordered)
        conv11 = self.conv11(conv10)
        return conv11

    def copy_params_from_vgg16(self):
        features = [
            self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]


        vgg16 = torchvision.models.vgg16(pretrained=True)
        #state_dict = torch.load(vgg_model_file)
        #vgg16.load_state_dict(state_dict)


        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
