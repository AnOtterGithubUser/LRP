import torch
import torch.nn as nn
from copy import deepcopy


class LRPLinear(nn.Linear):

    def forward(self, x):
        self.activations = x.detach().data
        if self.activations.dim() == 1:
            self.activations = self.activations.view(1, -1)
        return super(LRPLinear, self).forward(x)

    def backward_relevance(self, R, rule, l=0., h=1.):
        weights = self.weight.detach().data
        if rule == "w2":
            Z = torch.sum(weights * weights, dim=1)
            S = R / Z
            R = torch.mm(weights * weights, S)
        elif rule == "z+":
            V = weights
            V[V<=0] = 0
            Z = torch.mm(self.activations, V.transpose(1, 0)) + 1e-9
            S = R / Z
            C = torch.mm(S, V)
            R = C * self.activations
        elif rule == "zbeta":
            V, U = deepcopy(weights), deepcopy(weights)
            V[V<=0] = 0
            U[U>=0] = 0
            L, H = self.activations*0 + l, self.activations*0 + h
            Z = torch.mm(self.activations, weights.transpose(1, 0)) - torch.mm(L, V.transpose(1, 0)) - torch.mm(H, U.transpose(1, 0)) + 1e-9
            S = R / Z
            R = self.activations * torch.mm(S, weights) - L * torch.mm(S, V) - H * torch.mm(S, U)
        else:
            raise ValueError("rule %s is not valid (options: 'w2', 'z+', 'zbeta')" % rule)
        return R


class LRPReLU(nn.ReLU):

    def backward_relevance(self, R, rule):
        return R


class LRPConv2d(nn.Conv2d):

    def backward_relevance(self, R, rule):
        raise NotImplementedError
        
class LRPMaxPool2d(nn.MaxPool2d):
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=True, ceil_mode=False):
        super(LRPMaxPool2d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        
    def forward(self, x):
        y, indices = super(LRPMaxPool2d, self).forward(x)
        self.binary_mask = torch.zeros(x.size())
        self.binary_mask[indices] = 1
        return y
    
    def backward_relevance(self, R):
        return self.binary_mask * R
