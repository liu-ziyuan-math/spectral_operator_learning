# Following code of func. count_params and LpLoss are given by \
# https://github.com/zongyi-li/fourier_neural_operator/blob/master/utilities3.py


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import operator
#from functools import partial


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

    
    
class my_plt(object):
    def __init__(self, model, x_data, y_data, mesh, loss_fun, label='OPNO', clr='r'):
        super(my_plt, self).__init__()

        font1 = {'size': 23}
        with torch.no_grad():
            self.x, self.y = x_data.to(next(model.parameters()).device), y_data.detach().to('cpu')
            self.yy = model(self.x).cpu().reshape(x_data.shape[0], -1)
            self.mesh = mesh
            self.label = label
            self.lossfun = loss_fun
            self.clr = clr

    def ppt(self, j):
        # j += 1
        plt.cla()
        plt.scatter(self.mesh, self.yy[j, :], color=self.clr, s=200, alpha=0.75, label=self.label)
        # plt.plot(-torch.cos(torch.linspace(0, np.pi, Nx)), yy[j, :].detach().to('cpu'),color='r')
        plt.plot(self.mesh, self.x[j, :, 0].cpu(), ':', label='$u_0$', linewidth=5)
        plt.plot(self.mesh, self.y[j, :], color='b', label='$u_1$ ref',
                    linewidth=2)  # - y[j, :]
        print(self.lossfun(self.yy[j:j+1, :], self.y[j:j+1, :]))

        plt.tick_params(labelsize=40)
        plt.legend(fontsize=40)
