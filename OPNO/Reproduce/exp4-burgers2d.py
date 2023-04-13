"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
The part of SpectralConv2d is written by Zongyi Li(1) and we
"""

sub_list = [2**2, 2**1, 2**0]
sub = sub_list[0] ## modify this parameter to complete the whole example

data_PATH = '../data/burgers2d.mat'
file_name = 'opno2d'+str(sub)
result_PATH = '../model/' + file_name + '.pkl'

import sys
sys.path.append("..")
from OPNO2d import *

torch.manual_seed(0)
np.random.seed(0)
#if device == torch.device('cuda'):
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


epochs = 3000
batch_size = 20
learning_rate = 0.001
step_size = 300  # for StepLR
gamma = 0.5  # for StepLR
weight_decay = 1e-4
train_size, test_size = 1000, 100

degree = 16
width = 24


torch.manual_seed(0)
np.random.seed(0)
#if device == torch.device('cuda'):
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs)
print('weight_decay', weight_decay, 'width', width, 'degree', degree, 'sub', sub)

## main
model = OPNO2d(degree, degree, width).to(device)

print('supervising data loaded! PATH = ' + data_PATH)
raw_data = h5py.File(data_PATH, 'r')
x_data, y_data = raw_data['u_cgl'][..., 0], raw_data['u_cgl']
x_data, y_data = torch.tensor(x_data[..., ::sub, ::sub]), torch.tensor(y_data[..., ::sub, ::sub, :])
y_data = y_data[..., (2, 6, 10)]
data_size, Nx, Ny = x_data.shape
print('data size = ', data_size, 'Nx = ', Nx, '*', Ny)
x_data, y_data = x_data.reshape(-1, Nx, Ny, 1), y_data.reshape(-1, Nx, Ny, 3)

grid_x = -torch.cos(torch.linspace(0, np.pi, Nx, dtype=torch.float64)).view(1, Nx, 1, 1)
grid_y = -torch.cos(torch.linspace(0, np.pi, Ny, dtype=torch.float64)).view(1, 1, Ny, 1)

x_data = torch.cat([x_data.view(data_size, Nx, Ny, 1)
                       , grid_x.repeat(data_size, 1, Ny, 1), grid_y.repeat(data_size, Nx, 1, 1)], dim=-1)
x_data = x_data.view(data_size, Nx, Ny, 3)
print("data_shape", x_data.shape, y_data.shape)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[:train_size, ...], y_data[:train_size, ...]), batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[-test_size:, ...], y_data[-test_size:, ...]), batch_size=batch_size,
    shuffle=False)

print('model parameters number =', count_params(model))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
from Adam import Adam
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_list, loss_list = [], []

myloss = LpLoss(size_average=False)


if epochs == 0: #load model
    print('model:'+result_PATH+' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    #peer_err = loader['test_err']

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
    if (ep + 1) % 1 == 0:
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
              train_mse, train_l2, test_l2)

x, y = x_data[-test_size:, ...].to(device), y_data[-test_size:, ...]
with torch.no_grad():
    yy = model(x).reshape(test_size, Nx, Nx, 3).cpu()
j = -1

p = ch.cheb_partial(yy, 1)
p = p[:, (0, -1), :, :].reshape(test_size, -1)
ans1, _ = torch.max(torch.abs(p), dim=1)
p = ch.cheb_partial(yy, 2)
p = p[:, :, (0, -1), :].reshape(test_size, -1)
ans2, _ = torch.max(torch.abs(p), dim=1)

ans, _ = torch.max(torch.vstack([ans1, ans2]), dim=0)

print(torch.mean(ans))

if epochs >= 3000:
    torch.save({
        'model':model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'degree': degree,
        'loss_list': loss_list, 'train_list': train_list
    }, result_PATH)


    ## training
    #file_name = os.path.basename(__file__).split('.')[0]

