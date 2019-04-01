import torch
import torch.nn as nn
from copy import deepcopy

#TODO: replace the syntax V[V<0] = 0 by torch.maximum(0)

class LRPLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super(LRPLinear, self).__init__(*args, **kwargs)

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


class LRPConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(LRPConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        return super(LRPConv2d, self).forward(x)

    def backward_relevance(self, R, rule):
        raise NotImplementedError


class LRPMaxPool2d(nn.MaxPool2d):

    def __init__(self, *args, **kwargs):
        kwargs["return_indices"] = True
        super(LRPMaxPool2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.activations = x.detach().data
        y, self.indices = super(LRPMaxPool2d, self).forward(x)
        return y
    
    def backward_relevance(self, R):
        mask = torch.zeros(self.activations.numel())
        mask[self.indices] = R
        mask = mask.view(self.activations.size())
        return mask
