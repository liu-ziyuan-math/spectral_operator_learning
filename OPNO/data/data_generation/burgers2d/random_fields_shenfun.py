'''
Generating the Gauss Random Field N(0, \sigma^2(-Delta + \tau^2 I)^{-\alpha}) on the uniform
 discrete grids of [-1, 1], and then sample the eigenfunction on CGL points
'''

"""
u = \sum_{k=0}^N \sum_{l=0}^N coef_kl exp(ik pi/2 x) exp(il pi/2 y)
where coef_kl = [(k pi/2)^2 + (l pi/2)^2 + tau^2]^(alpha/2)0
"""
#import torch
from math import pi

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from shenfun import *
import sympy as sp

from shenfun import comm, FunctionSpace, TensorProductSpace
from timeit import default_timer
from mpl_toolkits.mplot3d import Axes3D


class GaussianRF(object):

    def __init__(self, dim, s, N=None, alpha=2, tau=3, sigma=None, bc=None):

        self.dim = dim
        self.s = s
        self.bc = bc
        self.N = s if N == None else N

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = s//2

        if dim == 1:
            #k = torch.arange(start=0, end=k_max, step=1, device=device)
            k = np.arange(0, k_max)

            self.sqrt_eig = sigma*(((pi/2*2)**2 * k**2) + tau**2)**(-alpha/2.0)
            self.C = FunctionSpace(self.N, 'C', quad='GL', bc=bc)
            x_cgl = self.C.mesh().reshape(-1, 1)

        elif dim == 2:
            k = np.arange(0, k_max).reshape(-1, 1)
            k_y = k.T

            self.sqrt_eig = sigma*(((pi/2*2)**2 * (k**2 + k_y**2)) + tau**2)**(-alpha/2.0)
            # self.sqrt_eig[0,0] = 0.0
            CX = FunctionSpace(self.N, 'C', quad='GL', bc=bc)
            CY = FunctionSpace(self.N, 'C', quad='GL', bc=bc)
            self.C = TensorProductSpace(comm, (CX, CY))
            x_cgl = CX.mesh().reshape(-1, 1)

        # self.EXP = np.exp(2*1j * np.pi * x_cgl @ k.reshape(1, -1))
        # [0, 2] ~> [0, 2pi]
        k = k.reshape(1, -1)
        if bc['left'][0] == 'D' and bc['right'][0] == 'D':# sum of Bsin
            self.type = 'dirichlet'
            self.EXP = np.sin(pi/2 * (x_cgl+1) @ k)# [-1, 1] -> [0, 2]

        elif bc['left'] == ('N',0) and bc['right'] == ('N',0): # sum of Acos
            self.type = 'neumann'
            self.EXP = np.cos(pi/2 * (x_cgl+1) @ k)# [-1, 1] -> [0, 2]
        else:
            raise("Currently unsupported B.C. for u0")

    def sample(self, ploting=False):

        coef = np.random.randn(*self.sqrt_eig.shape)*self.sqrt_eig

        # tranform : u(x) = \sum_k uh(k) EXP[k, x]
        if self.dim == 1:
            print(coef.shape, self.EXP.shape)
            u0_cgl = np.einsum("i, xi->x", coef, self.EXP)
        elif self.dim == 2:
            u0_cgl = np.einsum("ij, xi, yj->xy", coef, self.EXP, self.EXP)

        if self.type=='dirichlet' and (self.bc['left'][1] != 0 or self.bc['right'][1] != 0):
            a, b = self.bc['left'][1], self.bc['right'][1]
            u0_cgl += (b-a)/2 * self.C.mesh() + (b+a)/2

        if ploting:
            if self.dim==1:
                plt.plot(self.C.mesh(),u0.backward())
            else:
                # fig, ax = plt.subplots()
                # X, Y = np.meshgrid(*self.C.mesh())
                # u0 = Function(self.C)
                # u0[:] = self.C.forward(u0_cgl)
                # cs = ax.contourf(X, Y, u0.backward(), cmap=plt.get_cmap('Spectral'))
                # cbar = fig.colorbar(cs)
                # plt.show()

                X, Y = np.meshgrid(*self.C.mesh())
                fig = plt.figure()
                plt.cla()
                plt.pcolor(X, Y, u0_cgl, shading='auto', cmap="jet")
                print(np.max(np.abs(u0_cgl)))
                plt.colorbar()
                plt.show()

        return u0_cgl

if __name__ == "__main__":
    import chebypack as ch

    GRF = GaussianRF(2, 256, alpha=2.5, tau=7, sigma=7**1.5, bc={'left': ('N', 0), 'right': ('N', 0)})
    GRF = GaussianRF(2, 256, alpha=2, tau=4, sigma=9, bc={'left': ('N', 0), 'right': ('N', 0)})
    u0 = GRF.sample(ploting=1)
