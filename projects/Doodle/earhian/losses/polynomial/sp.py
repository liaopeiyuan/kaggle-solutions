import torch
import torch.nn as nn
import torch.autograd as ag

from losses.polynomial.divide_conquer import divide_and_conquer
from losses.polynomial.multiplication import Multiplication
from losses.polynomial.grad import d_logS_d_expX


class LogSumExp(nn.Module):
    def __init__(self, k, p=None, thresh=1e-5):
        super(LogSumExp, self).__init__()
        self.k = k
        self.p = int(1 + 0.2 * k) if p is None else p
        self.mul = Multiplication(self.k + self.p - 1)
        self.thresh = thresh

        self.register_buffer('grad_k', torch.Tensor(0))
        self.register_buffer('grad_km1', torch.Tensor(0))

        self.buffers = (self.grad_km1, self.grad_k)

    def forward(self, x):
        f = LogSumExp_F(self.k, self.p, self.thresh, self.mul, self.buffers)
        return f(x)


class LogSumExp_F(ag.Function):
    def __init__(self, k, p, thresh, mul, buffers):
        self.k = k
        self.p = p
        self.mul = mul
        self.thresh = thresh

        # unpack buffers
        self.grad_km1, self.grad_k = buffers

    def forward(self, x):
        """
        Returns a matrix of size (2, n_samples) with sigma_{k-1} and sigma_{k}
        for each sample of the mini-batch.
        """
        self.save_for_backward(x)

        # number of samples and number of coefficients to compute
        n_s = x.size(0)
        kp = self.k + self.p - 1

        assert kp <= x.size(1)

        # clone to allow in-place operations
        x = x.clone()

        # pre-compute normalization
        x_summed = x.sum(1)

        # invert in log-space
        x.t_().mul_(-1)

        # initialize polynomials (in log-space)
        x = [x, x.clone().fill_(0)]

        # polynomial multiplications
        log_res = divide_and_conquer(x, kp, mul=self.mul)

        # re-normalize
        coeff = log_res + x_summed[None, :]

        # avoid broadcasting issues (in particular if n_s = 1)
        coeff = coeff.view(kp + 1, n_s)

        # save all coeff for backward
        self.saved_coeff = coeff

        return coeff[self.k - 1: self.k + 1]

    def backward(self, grad_sk):
        """
        Compute backward pass of LogSumExp.
        Python variables with an upper case first letter are in
        log-space, other are in standard space.
        """

        # tensors from forward pass
        X, = self.saved_tensors
        S = self.saved_coeff

        # extend to shape (self.k + 1, n_samples, n_classes) for backward
        S = S.unsqueeze(2).expand(S.size(0), X.size(0), X.size(1))

        # compute gradients for coeff of degree k and k - 1
        self.grad_km1 = d_logS_d_expX(S, X, self.k - 1, self.p, self.grad_km1, self.thresh)
        self.grad_k = d_logS_d_expX(S, X, self.k, self.p, self.grad_k, self.thresh)

        # chain rule: combine with incoming gradients (broadcast to all classes on third dim)
        grad_x = grad_sk[0, :, None] * self.grad_km1 + grad_sk[1, :, None] * self.grad_k

        return grad_x


def log_sum_exp(x):
    """
    Compute log(sum(exp(x), 1)) in a numerically stable way.
    Assumes x is 2d.
    """
    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))
