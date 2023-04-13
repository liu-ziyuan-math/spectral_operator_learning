# This is the code for reproducing the Experiment 2: Heat diffusion equation with Robin BCs
# In this experiment, we tested OPNO with basis functions satisfying Robin BCs with different
# size of CGL discretization points.
# And this program is for the comparison with numerical method.

sub = 1
batch_size = 20
train_size_list = [100, 300, 1000, 3000, 10000, 30000, 100000]
train_size = 10000 # The size of training set is the control variable in this sub-example.

data_PATH = '../data/heat_robinN256.mat'
file_name = 'opno-robin'+str(sub)
result_PATH = '../model/' + file_name + '.pkl'

import sys
sys.path.append("..")
from OPNO1d_robin import *

torch.manual_seed(0)
np.random.seed(0)
if device == torch.device('cuda'):
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True


epochs = 5000 if train_size <= 10000 else 6000 # To mitigate underfitting, epochs set as 6000 when train_size is large
learning_rate = 0.001
step_size = 500  # for StepLR
gamma = 0.5  # for StepLR
weight_decay = 1e-4 if train_size <= 10000 else 0 # To mitigate underfitting, weight_decay set as 0 when train_size is large
# train_size, test_size = 1000, 1000
test_size = 1000

degree = 40
width = 50

print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs)
print('weight_decay', weight_decay, 'width', width, 'degree', degree, 'sub', sub)

## main
model = OPNO(degree, width).to(device)

print('supervising data loaded! PATH = ' + data_PATH)
raw_data = h5py.File(data_PATH, 'r')
x_data, y_data = np.array(raw_data['u0_cgl']), np.array(raw_data['u1_cgl'])
x_data, y_data = torch.tensor(x_data[:, ::sub]), torch.tensor(y_data[:, ::sub])
data_size, Nx = x_data.shape
print('data size = ', data_size, 'Nx = ', Nx)

grid = -torch.cos(torch.linspace(0, np.pi, Nx, dtype=torch.float64)).reshape(1, Nx, 1)
# Inserting the information of grids is not necessary or even not helpful
# but we keep it for the comparison with other models

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
    if (ep+1) % 1 == 0:
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
              train_mse, train_l2, test_l2)

xx, y = x_data[-test_size:, :, :].to(device), y_data[-test_size:, :]
with torch.no_grad():
    yy = model(xx).reshape(test_size, -1).cpu()

ydiff = ch.cheb_partial(yy, -1)
err_bc = ydiff[:, (0, -1)] * torch.tensor([1, -1]) + yy[:, (0, -1)]
ans, _ = torch.max(torch.abs(err_bc), dim=1)
print('bc_err:', torch.mean(ans))

show = my_plt(model, x_data[-test_size:, ...], y_data[-test_size:, ...], -torch.cos(torch.linspace(0, np.pi, Nx)), myloss)
j = -1

plt.figure(figsize=(14, 10))
j += 1
show.ppt(j)
plt.show()

if epochs >= 3000:
    torch.save({
        'model':model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'degree': degree,
        'loss_list': loss_list, 'train_list': train_list
    }, result_PATH)

