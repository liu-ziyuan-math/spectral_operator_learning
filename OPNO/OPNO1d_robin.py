"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn / liuziyuan@pku.edu.cn
For a middle- or left-engine rear PseudoSpectra please refer to OPNO1d_dirichlet.py (self.central)
### Robin boundary condition
"""

from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
import h5py
import chebypack as ch
import matplotlib

x2phi = functools.partial(ch.Wrapper, [ch.dct, ch.cmp_robin])
phi2x = functools.partial(ch.Wrapper, [ch.icmp_robin, ch.idct])
idctn = functools.partial(ch.Wrapper, [ch.idct])
dctn = functools.partial(ch.Wrapper, [ch.dct])
device = torch.device('cuda')

class PseudoSpectra(nn.Module):
    def __init__(self, in_channels, out_channels, degree, bandwidth):
        super(PseudoSpectra, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.bandwidth = bandwidth

        self.scale = (2 / (in_channels + out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(degree, in_channels, out_channels, bandwidth, dtype=torch.float64))

    def quasi_diag(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        # x : (batches, nx, features)
        batch_size, width, Nx = u.shape

        b = dctn(u, -1)

        out = torch.zeros(batch_size, self.out_channels, Nx, device=u.device, dtype=torch.float64)
        out[..., :self.degree] = self.quasi_diag(b[..., :self.degree+2], self.weights)

        u = phi2x(out, -1)
        return u


class OPNO(nn.Module):
    def __init__(self, degree, width):
        super(OPNO, self).__init__()

        self.degree = degree
        self.width = width

        self.conv0 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv1 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv2 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv3 = PseudoSpectra(self.width, self.width, self.degree, 3)
        self.conv4 = PseudoSpectra(self.width, self.width, self.degree, 3)

        self.convl = PseudoSpectra(2, self.width-2, self.degree, 3)

        self.w0 = nn.Conv1d(self.width, self.width, 1).double()#better
        self.w1 = nn.Conv1d(self.width, self.width, 1).double()
        self.w2 = nn.Conv1d(self.width, self.width, 1).double()
        self.w3 = nn.Conv1d(self.width, self.width, 1).double()

        self.fc1 = nn.Linear(self.width, 128).double()
        self.fc2 = nn.Linear(128, 1).double()

    def acti(self, x):
        return F.gelu(x)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = torch.cat([x, self.acti(self.convl(x))], dim=1)

        x = x+self.acti(self.w0(x) + self.conv0(x))

        x = x+self.acti(self.w1(x) + self.conv1(x))

        x = x+self.acti(self.w2(x) + self.conv2(x))

        x = x+self.acti(self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.acti(x)
        x = self.fc2(x)
        x = phi2x(x2phi(x, -2), -2)

        return x

if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    ### parameters settings
    data_PATH = '../data/heat_robin100k.mat'
    epochs = 6000
    batch_size = 20
    learning_rate = 0.001
    step_size = 500  # for StepLR
    gamma = 0.5  # for StepLR
    weight_decay = 1e-4 * 0
    sub = 2**5
    train_size, test_size = 30000, 1000

    degree = 40
    width = 50

    print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs)
    print('weight_decay', weight_decay, 'width', width, 'degree', degree, 'sub', sub)

    ## main
    model = OPNO(degree, width).to(device)

    print('supervising data loaded! PATH = ' + data_PATH)
    # raw_data = loadmat(data_PATH)
    raw_data = h5py.File(data_PATH, 'r')
    x_data, y_data = raw_data['u0_cgl'], raw_data['u1_cgl']
    x_data, y_data = x_data[:, ::sub], y_data[:, ::sub]
    x_data, y_data = torch.tensor(x_data), torch.tensor(y_data)

    data_size, Nx = x_data.shape
    print('data size = ', data_size, 'training size = ', train_size, 'test size = ', test_size, 'Nx = ', Nx)

    grid = -torch.cos(torch.linspace(0, np.pi, Nx, dtype=torch.float64)).reshape(1, Nx, 1)
    x_data = torch.cat([x_data.reshape(data_size,Nx,1)
                        , grid.repeat(data_size,1,1)], dim=2)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_data[:train_size, :, :], y_data[:train_size, :]), batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_data[-test_size:, :, :], y_data[-test_size:, :]), batch_size=batch_size,
        shuffle=False)

    print('model parameters number =', count_params(model))

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            # mse.backward()
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
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
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= train_size
        test_l2 /= test_size

        train_list.append(train_l2)
        loss_list.append(test_l2)

        t2 = default_timer()
        if (ep+1) % 10 == 0 or (ep < 5):
            print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
                  train_mse, train_l2, test_l2)

    xx, y = x_data[-test_size:, :, :].to(device), y_data[-test_size:, :]
    with torch.no_grad():
        yy = model(xx).reshape(test_size, -1).cpu()

    ydiff = ch.cheb_partial(yy, -1)
    err_bc = ydiff[:, (0, -1)]*torch.tensor([1,-1]) + yy[:, (0, -1)]
    ans, _ = torch.max(torch.abs(err_bc), dim=1)
    print('bc_err:', torch.mean(ans))

    peer_loss = LpLoss(reduction=False)
    test_err = peer_loss(yy.view(y.shape[0], -1), y.view(y.shape[0], -1))
    #print(test_err)
