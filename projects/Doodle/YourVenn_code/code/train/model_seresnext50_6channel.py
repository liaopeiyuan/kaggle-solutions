
import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'

from net.imagenet_pretrain_model.senet import *
from utils import *

BatchNorm2d = nn.BatchNorm2d

###########################################################################################3
class Net(nn.Module):
    def load_pretrain(self, pretrain_file, from_3channel = True):
        #raise NotImplementedError
        #self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())

        for key in keys:

            if from_3channel and (r'conv1.0.weight' in key or r'encoder.layer0.conv1.weight' in key):
                print(key)
                state_dict[key] = state_dict[key]
            else:
                # print(key)
                # st_key = r'module.'+key
                # if from_3channel:
                # state_dict[key] = pretrain_state_dict[st_key]
                # else:
                state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=340, is_cosine_linear = False):
        super(Net,self).__init__()

        self.encoder  = se_resnext50_32x4d()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder.layer0.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)

        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.fc1 = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(inplace=True))

        self.logit = nn.Linear(512, num_class)



    def forward(self, x):
        batch_size,C,H,W = x.shape
        mean=[0.5, 0.5, 0.5,0.5, 0.5, 0.5] #rgb
        std =[0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

        x = torch.cat([
            (x[:,[0]]-mean[0])/std[0],
            (x[:,[1]]-mean[1])/std[1],
            (x[:,[2]]-mean[2])/std[2],
            (x[:, [3]] - mean[3]) / std[3],
            (x[:, [4]] - mean[4]) / std[4],
            (x[:, [5]] - mean[5]) / std[5],
        ],1)

        x = self.conv1(x) #; print('e1',x.size())
        x = self.conv2(x) #; print('e2',x.size())
        x = self.conv3(x) #; print('e3',x.size())
        x = self.conv4(x) #; print('e4',x.size())
        x = self.conv5(x) #; print('e5',x.size())

        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size,-1)
        x = F.dropout(x, p=0.50, training=self.training)
        x = self.fc1(x)
        x = F.dropout(x, p=0.50, training=self.training)

        logit = self.logit(x)
        return logit

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

### run ##############################################################################
def run_check_net():
    batch_size = 32
    C,H,W = 6, 64, 64
    num_class = 340

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (num_class,   batch_size).astype(np.float32)

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).long().cuda()

    input = to_var(input)
    truth = to_var(truth)
    #
    #---
    # criterion = softmax_cross_entropy_criterion

    net = Net(num_class).cuda()
    # net = torch.nn.DataParallel(net).cuda()
    net.load_pretrain(r'/data2/shentao/DATA/Kaggle_draw_Venn/output_r50_144_ft/checkpoint/00006000_model.pth',from_3channel=True)



    # net.set_mode('train')
    # print(net)
    ## exit(0)

    # net.load_pretrain('/media/st/SSD02/Projects/Kaggle_draw/models/resnet34-fold0/checkpoint/00006000_model.pth')

    logit = net.forward(input)
    # loss  = criterion(logit, truth)
    # precision, top = metric(logit, truth)
    #
    # print('loss    : %0.8f  '%(loss.data.cpu().numpy()))
    # print('correct :(%0.8f ) %0.8f  %0.8f '%(precision.data.cpu().numpy(), top[0].data.cpu().numpy(),top[-1].data.cpu().numpy()))
    # print('')
    #
    # # dummy sgd to see if it can converge ...
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
    #                   lr=0.1, momentum=0.9, weight_decay=0.0001)
    #
    # i=0
    # optimizer.zero_grad()
    # print('        loss  | prec      top      ')
    # print('[iter ]       |           1  ... k ')
    # print('-------------------------------------')
    # while i<=500:
    #
    #     logit   = net(input)
    #     loss    = criterion(logit, truth)
    #     precision, top = metric(logit, truth)
    #
    #     loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    #     optimizer.step()
    #     optimizer.zero_grad()
    #
    #     if i%20==0:
    #         print('[%05d] %0.3f | ( %0.3f ) %0.3f  %0.3f'%(
    #             i, loss.data.cpu().numpy(),precision.data.cpu().numpy(), top[0].data.cpu().numpy(),top[-1].data.cpu().numpy(),
    #         ))
    #     i = i+1

########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_net()
    print( 'sucessful!'))