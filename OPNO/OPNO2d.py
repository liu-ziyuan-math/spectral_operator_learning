"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
T
"""
import os
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from copy import deepcopy
import h5py
from scipy.io import loadmat
import chebypack as ch
import functools

import matplotlib
x2phi = functools.partial(ch.Wrapper, [ch.dct, ch.cmp_neumann])
phi2x = functools.partial(ch.Wrapper, [ch.icmp_neumann, ch.idct])
idctn = functools.partial(ch.Wrapper, [ch.idct])
dctn = functools.partial(ch.Wrapper, [ch.dct])
device = torch.device('cuda')


class PseudoSpectra2d(nn.Module):
    def __init__(self, in_channels, out_channels, degree1, degree2, bandwidth):
        super(PseudoSpectra2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree1 = degree1
        self.degree2 = degree2
        self.bandwidth = bandwidth

        self.scale = 2 / (in_channels + out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels*bandwidth*bandwidth, out_channels, degree1*degree2, dtype=torch.float64))

        # self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth), padding=(self.bandwidth-1)//2)
        self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth))

    def quasi_diag_mul2d(self, input, weights):
        xpad = self.unfold(input)
        return torch.einsum("bix, iox->box", xpad, weights)
        # return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        batch_size, width, Nx, Ny = u.shape

        a = dctn(u, [-1, -2])

        b = torch.zeros(batch_size, self.out_channels, Nx, Ny, device=u.device, dtype=torch.float64)
        b[..., :self.degree1, :self.degree2] = \
            self.quasi_diag_mul2d(a[..., :self.degree1+2, :self.degree2+2], self.weights).reshape(
                batch_size, self.out_channels, self.degree1, self.degree2)

        u = phi2x(b, [-1, -2])
        return u


class OPNO2d(nn.Module):
    def __init__(self, degree1, degree2, width):
        super(OPNO2d, self).__init__()

        self.degree1 = degree1
        self.degree2 = degree2
        self.width = width

        self.conv0 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)
        self.conv1 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)
        self.conv2 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)
        self.conv3 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)
        # self.conv4 = PseudoSpectra2d(self.width, self.width, self.degree1, self.degree2, 3)

        self.convl = PseudoSpectra2d(3, self.width-3, self.degree1, self.degree2, 3)

        # self.w0 = Conv2d_1x1(self.width, self.width)
        # self.w1 = Conv2d_1x1(self.width, self.width)
        # self.w2 = Conv2d_1x1(self.width, self.width)
        # self.w3 = Conv2d_1x1(self.width, self.width)

        self.w0 = nn.Conv2d(self.width, self.width, 1).double()
        self.w1 = nn.Conv2d(self.width, self.width, 1).double()
        self.w2 = nn.Conv2d(self.width, self.width, 1).double()
        self.w3 = nn.Conv2d(self.width, self.width, 1).double()

        self.fc1 = nn.Linear(self.width, 128).double()
        self.fc2 = nn.Linear(128, 3).double()

    def acti(self, x):
        return F.gelu(x)

    def forward(self, x):
        # x : (batches, nx, ny, [Einc(x, y), cnt(x, y), x, y])

        x = x.permute(0, 3, 1, 2)

        x = torch.cat([x, self.acti(self.convl(x))], dim=1)

        x = x+self.acti(self.w0(x) + self.conv0(x))

        x = x+self.acti(self.w1(x) + self.conv1(x))

        x = x+self.acti(self.w2(x) + self.conv2(x))

        x = x+self.acti(self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x(x2phi(x, [1, 2]), [1, 2])

        return x

if __name__ == '__main__':

    #### parameters settings

    torch.manual_seed(0)
    np.random.seed(0)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    ## training
    data_PATH = './data/burgers2d.mat'

    epochs = 5
    batch_size = 20
    learning_rate = 0.001
    step_size = 300  # for StepLR
    gamma = 0.5  # for StepLR
    weight_decay = 1e-4
    sub = 2**2
    train_size, test_size = 1000, 100

    degree = 16
    width = 24

    print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs)
    print('weight_decay', weight_decay, 'width', width, 'degree', degree, 'sub', sub)

    load_data = True

    ## main

    model = OPNO2d(degree, degree, width).to(device)

    if load_data:
        print('supervising data loaded! PATH = ' + data_PATH)
        #raw_data = loadmat(data_PATH)
        raw_data = h5py.File(data_PATH, 'r')

        x_data, y_data = raw_data['u_cgl'][..., 0], raw_data['u_cgl']

        x_data, y_data = torch.tensor(x_data[..., ::sub, ::sub]), torch.tensor(y_data[..., ::sub, ::sub, :])
        y_data = y_data[..., (2, 6, 10)]
        data_size, Nx, Ny = x_data.shape
        print('data size = ', data_size, 'Nx = ', Nx, '*', Ny)
        x_data, y_data = x_data.reshape(-1, Nx, Ny, 1), y_data.reshape(-1, Nx, Ny, 3)

        grid_x = -torch.cos(torch.linspace(0, np.pi, Nx, dtype=torch.float64)).view(1, Nx, 1, 1)
        grid_y = -torch.cos(torch.linspace(0, np.pi, Ny, dtype=torch.float64)).view(1, 1, Ny, 1)

        x_data = torch.cat([x_data.view(data_size,Nx,Ny,1)
                               , grid_x.repeat(data_size,1,Ny,1), grid_y.repeat(data_size,Nx,1,1)], dim=-1)
        x_data = x_data.view(data_size, Nx, Ny, 3)
        print("data_shape", x_data.shape, y_data.shape)


        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_data[:train_size, ...], y_data[:train_size, ...]), batch_size=batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_data[-test_size:, ...], y_data[-test_size:, ...]), batch_size=batch_size,
            shuffle=False)

    print('model parameters number =', count_params(model))
    from Adam import Adam
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_list, loss_list = [], []

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out.reshape(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            # mse.backward()
            l2 = myloss(out.reshape(batch_size, -1), y.view(batch_size, -1))
            l2.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                test_l2 += myloss(out.reshape(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= train_size
        test_l2 /= test_size

        train_list.append(train_l2)
        loss_list.append(test_l2)

        t2 = default_timer()
        if (ep+1) % 1 == 0:
            print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
                  train_mse, train_l2, test_l2)
