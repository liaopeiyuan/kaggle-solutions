from common import *
from net.imagenet_pretrain_model.resnet import *




#<todo> bce, focal loss, direct top 5 sugorate
#<todo> 'smooth loss functions for deep top-k classification' - Leonard Berrada, Andrew Zisserman, M. Pawan Kumar
#       https://arxiv.org/abs/1802.07595



def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss


def metric(logit, truth, is_average=True):

    with torch.no_grad():
        prob = F.softmax(logit, 1)
        value, top = prob.topk(3, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        if is_average==True:
            # top-3 accuracy
            correct = correct.float().sum(0, keepdim=False)
            correct = correct/len(truth)

            top = [correct[0], correct[0]+correct[1], correct[0]+correct[1]+correct[2]]
            precision = correct[0]/1 + correct[1]/2 + correct[2]/3
            return precision, top

        else:
            return correct



###########################################################################################3

class Net(nn.Module):

    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        #self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in []):
                continue
            # if key.startswith('conv1.0'):
            #     state_dict[key] = pretrain_state_dict[key.replace('conv1.0','conv1')]
            # if key.startswith('conv1.1'):
            #     state_dict[key] = pretrain_state_dict[key.replace('conv1.1','bn1')]

            # if 'resnet.conv1.' in key:
            #     state_dict[key] = pretrain_state_dict[key.replace('resnet.conv1.','conv1.')]
            # if 'resnet.bn1.' in key:
            #     state_dict[key] = pretrain_state_dict[key.replace('resnet.bn1.','bn1.')]
            if 'encoder1.0.' in key:
                state_dict[key] = pretrain_state_dict[key.replace('encoder1.0.','conv1.')]
                print(key)
            if 'encoder1.1.' in key:
                state_dict[key] = pretrain_state_dict[key.replace('encoder1.1.','bn1.')]
                print(key)

            if any(s in key for s in []):
                continue
            if 'resnet.layer0.' in key:
                state_dict[key] = pretrain_state_dict[key.replace('resnet.layer0.','layer0.')]
                print(key)
            if 'resnet.layer1.' in key:
                state_dict[key] = pretrain_state_dict[key.replace('resnet.layer1.','layer1.')]
                print(key)
            if 'resnet.layer2.' in key:
                state_dict[key] = pretrain_state_dict[key.replace('resnet.layer2.','layer2.')]
                print(key)
            if 'resnet.layer3.' in key:
                state_dict[key] = pretrain_state_dict[key.replace('resnet.layer3.','layer3.')]
                print(key)
            if 'resnet.layer4.' in key:
                 state_dict[key] = pretrain_state_dict[key.replace('resnet.layer4.','layer4.')]
                 print(key)

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=340):
        super(Net,self).__init__()
        self.resnet  = ResNet(BasicBlock, [3, 4, 6, 3],num_classes=1)

        # self.conv1 = nn.Sequential(
        #     self.resnet.conv1,
        #     self.resnet.bn1,
        #     self.resnet.relu,
        #     #self.resnet.maxpool,
        # )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.resnet.layer1,
        )
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        self.logit = nn.Linear(512, num_class)



    def forward(self, x):
        batch_size,C,H,W = x.shape
        mean=[0.485, 0.456, 0.406] #rgb
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x[:,[0]]-mean[0])/std[0],
            (x[:,[1]]-mean[1])/std[1],
            (x[:,[2]]-mean[2])/std[2],
        ],1)

        x = self.encoder1(x) #; print('e1',x.size())
        x = self.encoder2(x) #; print('e2',x.size())
        x = self.encoder3(x) #; print('e3',x.size())
        x = self.encoder4(x) #; print('e4',x.size())
        x = self.encoder5(x) #; print('e5',x.size())

        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size,-1)
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
    C,H,W = 3, 32, 32
    num_class = 5

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (num_class,   batch_size).astype(np.float32)

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).long().cuda()


    #---
    criterion = softmax_cross_entropy_criterion
    net = Net(num_class).cuda()
    net.set_mode('train')
    # print(net)
    ## exit(0)

    net.load_pretrain('../../models/resnet34-333f7ec4.pth')

    logit = net(input)
    loss  = criterion(logit, truth)
    precision, top = metric(logit, truth)

    print('loss    : %0.8f  '%(loss.item()))
    print('correct :(%0.8f ) %0.8f  %0.8f '%(precision.item(), top[0].item(),top[-1].item()))
    print('')



    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    print('        loss  | prec      top      ')
    print('[iter ]       |           1  ... k ')
    print('-------------------------------------')
    while i<=500:

        logit   = net(input)
        loss    = criterion(logit, truth)
        precision, top = metric(logit, truth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] %0.3f | ( %0.3f ) %0.3f  %0.3f'%(
                i, loss.item(),precision.item(), top[0].item(),top[-1].item(),
            ))
        i = i+1





########################################################################################
if __name__ == '__main__':
    print( 'calling main function ... ')

    run_check_net()

    print( 'sucessful!')