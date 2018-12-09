from common import *


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


## net  ######################################################################
class SaltNet(nn.Module):

    def __init__(self, ):
        super(SaltNet, self).__init__()

        self.down1 = nn.Sequential(
            ConvBn2d(  1,  64, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d( 64,  64, kernel_size=3, stride=1, padding=1 ),
        )
        self.down2 = nn.Sequential(
            ConvBn2d( 64, 128, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        self.down3 = nn.Sequential(
            ConvBn2d(128, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        self.down4 = nn.Sequential(
            ConvBn2d(256, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )
        self.down5 = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )

        self.same = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )

        self.up5 = nn.Sequential(
            ConvBn2d(1024,512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
        )

        self.up4 = nn.Sequential(
            ConvBn2d(1024,512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1 ),
        )
        self.up3 = nn.Sequential(
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1 ),
        )
        self.up2 = nn.Sequential(
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1 ),
        )
        self.up1 = nn.Sequential(
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1 ),
            ConvBn2d( 64,  64, kernel_size=3, stride=1, padding=1 ),
        )
        self.feature = nn.Sequential(
            ConvBn2d( 64,  64, kernel_size=1, stride=1, padding=0 ),
        )
        self.logit = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0 )




    def forward(self, input):

        down1 = self.down1(input)
        f     = F.max_pool2d(down1, kernel_size=2, stride=2)#, return_indices=True)
        down2 = self.down2(f)
        f     = F.max_pool2d(down2, kernel_size=2, stride=2)
        down3 = self.down3(f)
        f     = F.max_pool2d(down3, kernel_size=2, stride=2)
        down4 = self.down4(f)
        f     = F.max_pool2d(down4, kernel_size=2, stride=2)
        down5 = self.down5(f)
        f     = F.max_pool2d(down5, kernel_size=2, stride=2)

        f  = self.same(f)

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        #f = F.max_unpool2d(f, i4, kernel_size=2, stride=2)
        f = self.up5(torch.cat([down5, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up4(torch.cat([down4, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up3(torch.cat([down3, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up2(torch.cat([down2, f],1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up1(torch.cat([down1, f],1))

        f = self.feature(f)
        #f = F.dropout(f, p=0.5)
        logit = self.logit(f)

        return logit


    def criterion(self, logit, truth ):
        loss = FocalLoss2d()(logit, truth, type='sigmoid')
        return loss

    # def criterion(self,logit, truth):
    #     loss = F.binary_cross_entropy_with_logits(logit, truth)
    #     return loss

    def metric(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        #dice = dice_accuracy(prob, truth, threshold=threshold, is_average=True)
        dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        return dice

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

        else:
            raise NotImplementedError





### run ##############################################################################



def run_check_net():

    batch_size = 8
    C,H,W = 1, 128, 128

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (2,   (batch_size,C,H,W)).astype(np.float32)


    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()


    #---
    net = SaltNet().cuda()
    net.set_mode('train')

    logit = net(input)
    loss  = net.criterion(logit, truth)
    dice  = net.metric(logit, truth)

    print('loss : %0.8f'%loss.item())
    print('dice : %0.8f'%dice.item())
    print('')


    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    while i<=500:

        logit = net(input)
        loss  = net.criterion(logit, truth)
        dice  = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] loss, dice  :  %0.5f,%0.5f'%(i, loss.item(),dice.item()))
        i = i+1

'''
model_1.py: calling main function ... 
loss : 0.09594455
dice : 0.31696570

[00000] loss, dice  :  0.09594,0.31697
[00020] loss, dice  :  0.07522,0.56402
[00040] loss, dice  :  0.05924,0.67840
[00060] loss, dice  :  0.03638,0.84401
[00080] loss, dice  :  0.01344,0.98623
[00100] loss, dice  :  0.00313,0.99994
[00120] loss, dice  :  0.00112,1.00000
[00140] loss, dice  :  0.00063,1.00000
[00160] loss, dice  :  0.00044,1.00000
[00180] loss, dice  :  0.00034,1.00000
[00200] loss, dice  :  0.00028,1.00000
[00220] loss, dice  :  0.00023,1.00000
[00240] loss, dice  :  0.00020,1.00000
'''





########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')