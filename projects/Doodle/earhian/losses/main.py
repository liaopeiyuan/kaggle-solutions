import torch.nn as nn
from losses.svm import SmoothSVM


def get_loss_svm():
    loss = SmoothSVM(n_classes=340, k=3, tau=1.0, alpha=1.0)
    return loss