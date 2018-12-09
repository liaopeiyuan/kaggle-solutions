from common import *
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def metric(logit, truth, is_average=True):
    # with torch.no_grad():
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

def do_valid( net, valid_loader, criterion ):
    valid_num  = 0
    probs    = []
    truths   = []
    losses   = []
    corrects = []

    for input, truth, _ in valid_loader:
        input = input.cuda()
        truth = truth.cuda()

        input = to_var(input)
        truth = to_var(truth)

        logit   = net(input)
        prob    = F.softmax(logit,1)

        loss    = criterion(logit, truth, False)
        correct = metric(logit, truth, False)

        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())
        losses.append(loss.data.cpu().numpy())
        corrects.append(correct.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())


    assert(valid_num == len(valid_loader.sampler))
    #------------------------------------------------------
    prob    = np.concatenate(probs)
    correct = np.concatenate(corrects)
    truth   = np.concatenate(truths).astype(np.int32).reshape(-1,1)
    loss    = np.concatenate(losses)
    #---
    #top = np.argsort(-predict,1)[:,:3]

    loss    = loss.mean()
    correct = correct.mean(0)
    top = [correct[0], correct[0]+correct[1], correct[0]+correct[1]+correct[2]]
    precision = correct[0]/1 + correct[1]/2 + correct[2]/3

    #----
    valid_loss = np.array([
        loss, top[0], top[2], precision
    ])

    return valid_loss