from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy
from shenfun import *
from mpi4py_fft import fftw
from timeit import default_timer
#matplotlib.use('TkAgg')
from random_fields_shenfun import GaussianRF
import time
import h5py
import sys

# parameters for 2d Burgers' equation
visc = 0.001

# parameters for discretization
dt = 0.01
record_timestep = 0.1
end_time = 1
Nx = 200 + 1
bc={'left': ('N', 0), 'right': ('N', 0)}
# bc={'right': {'N':0}, 'left': {'N':0}}

# GRF = GaussianRF(1, Nx, alpha=3, tau=5, sigma=25, bc=bc)
GRF = GaussianRF(2, Nx, alpha=2, tau=4, sigma=4, bc=bc)
File_name = "burgers2d_v1e-3_" + f"_{num}.mat"
print(File_name, "start")


class Burgers2dIRK3(IRK3):
    def __init__(self, N=10, dt=0.01, visc=0.1, modplot=10, family='C', bc={'right': {'D':0, 'N':0}}):

        self.visc = visc
        self.N = N
        self.dt = dt
        self.modplot = modplot
        self.bc = bc
        # T = FunctionSpace(N, family, quad='GL', bc=self.bc)
        CX = FunctionSpace(N, family, quad='GL', bc=self.bc)
        CY = FunctionSpace(N, family, quad='GL', bc=self.bc)
        self.T = TensorProductSpace(comm, (CX, CY))

        # Functions to hold solution
        self.u_ = Array(self.T)
        self.u_hat = Function(self.T)

        IRK3.__init__(self, self.T)

    def LinearRHS(self, u, **par):
        return self.visc * div(grad(u))

    def NonlinearRHS(self, u, u_hat, rhs, **params):
        # -(u^2/2)_x
        rhs.fill(0)
        v = TestFunction(self.T)
        ub = u_hat.backward(padding_factor=1.5)  # Use padding for nonlinear terms
        uu = (-0.5*ub**2).forward()
        rhs[:] = inner(Dx(uu, 0, 1)+Dx(uu, 1, 1), v)
        # rhs[:] = inner(grad(uu), v)
        return rhs

    def update(self, u, u_hat, t, tstep, **par):
        pass
        if tstep % self.modplot == 0:
            # print(name,tstep // self.modplot)
            j = tstep // self.modplot
            u_cgl[i, ..., j] = u_hat.backward()
            u_unif[i, ..., j] = u_hat.eval(grids).reshape(Nx, Nx)

            # k = (tstep-1)//self.modplot
            # u = u_hat.backward()

            # X, Y = GRF.C.local_mesh(True)
            # plt.subplot(2,5,k+1)
            # plt.pcolor(X, Y, u, shading='auto', cmap="jet")
            # # # print(np.max(np.abs(u0.backward())))
            # plt.colorbar()
            # plt.show()
            """
            Ny = Nx
            grid_x = np.cos(np.linspace(0, np.pi, Nx))
            grid_y = np.cos(np.linspace(0, np.pi, Ny))
            nx = grid_x.reshape(-1)
            ny = grid_y.reshape(-1)
            X, Y = np.meshgrid(nx, ny)
            k = j
            plt.subplot(3, 4, k+1)
            plt.pcolor(X, Y, u_cgl[i, ..., k], cmap="jet")
            plt.colorbar()
            """

if __name__ == '__main__':

    modplot = int(record_timestep/dt)

    d = {'N': Nx,
         'dt': dt,
         'modplot': modplot,
         'family': 'C',
         'bc' : bc
        }

    sol = Burgers2dIRK3(**d)
    x_unif = np.linspace(-1, 1, Nx)
    X, Y = np.meshgrid(x_unif, x_unif.T)
    grids = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)])

    u_cgl, u_unif = np.zeros([num, Nx, Nx, 1+int(end_time/record_timestep)]), \
                      np.zeros([num, Nx, Nx, 1+int(end_time/record_timestep)])

    for i in range(num):
        time1 = default_timer()
        sol.u_.fill(0)
        sol.u_hat.fill(0)
        
        u_cgl[i, ..., 0] = sol.u_[:] = GRF.sample()
        """
        Ny = Nx
        grid_x = np.cos(np.linspace(0, np.pi, Nx))
        grid_y = np.cos(np.linspace(0, np.pi, Ny))
        nx = grid_x.reshape(-1)
        ny = grid_y.reshape(-1)
        X, Y = np.meshgrid(nx, ny)
        k = 0
        plt.subplot(3, 4, k+1)
        plt.pcolor(X, Y, u_cgl[i, ..., k], cmap="jet")
        plt.colorbar() 
        """
        sol.u_hat = sol.u_.forward(sol.u_hat)
        u_unif[i, ..., 0] = sol.u_hat.eval(grids).reshape(Nx, Nx)

        sol.u_hat = sol.solve(sol.u_, sol.u_hat, sol.dt, (0, end_time))
        # u_ = sol.up_hat.backward()

        time2 = default_timer()
        print(name, i, time2-time1, np.max(u_cgl[i, ..., 0]), np.max(u_cgl))

        file = h5py.File(File_name, "w")
        file.create_dataset("u_cgl", data=u_cgl)
        file.create_dataset("u_unif", data=u_unif)
        file.create_dataset("i", data=i)
        file.close()
