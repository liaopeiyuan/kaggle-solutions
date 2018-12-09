from common import *


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
## https://www.kaggle.com/kmader/quickdraw-simple-models
class Net(nn.Module):

    def __init__(self, num_class=340):
        super(Net,self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv1d(  3,  48, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d( 48, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(128, 256, num_layers=1, dropout=0, bidirectional=True, batch_first=True)
        #True  #False

        self.logit = nn.Linear(256*2, num_class)



    def forward(self, x, length):
        batch_size, T, dim = x.shape

        x = x.permute(0,2,1)
        x = self.encoder1(x) #; print('e1',x.size())
        x = self.encoder2(x) #; print('e2',x.size())
        x = self.encoder3(x) #; print('e3',x.size())
        x = x.permute(0,2,1)

        if 1:
            xx = pack_padded_sequence(x, length, batch_first=True)
            yy, (h, c) = self.lstm (xx)
            y, _ = pad_packed_sequence(yy, batch_first=True)
        else:
            y, (h, c) = self.lstm (x)

        #z = y[:,-1,:]  #last one
        #z = h.view(batch_size,-1)  #hidden (one direction)
        z = h.permute(1,0,2).contiguous().view(batch_size,-1)  #hidden (bi direction)
        #z = torch.cat((h[0],h[1]),1)


        logit = self.logit(z)
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

    dim = 3
    num_class = 10
    batch_size = 32
    length = np.random.randint(32,64,batch_size)
    length = np.sort(length)[::-1]
    length_max = length.max()

    truth = np.random.choice (num_class,   batch_size).astype(np.float32)
    sequence = [
        np.random.uniform(0,1, (length[i],dim)).astype(np.float32)
        for i in range(batch_size)
    ]
    ## pack to equal length
    input = np.zeros((batch_size, length_max, dim), np.float32)
    for i in range(batch_size):
        input[i, 0:length[i]] = sequence[i]

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).long().cuda()


    #---
    criterion = softmax_cross_entropy_criterion
    net = Net(num_class).cuda()
    net.set_mode('train')
    # print(net)
    ## exit(0)


    logit = net(input, length)
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

        logit = net(input, length)
        loss  = criterion(logit, truth)
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
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')