import functools
import chebypack as ch
def load_experiment_para(glob, para_list):
    default_para = {
        'epochs': 5000,
        'batch_size': 20,
        'learning_rate': 0.001,
        'step_size': 500,  # for StepLR
        'gamma': 0.5,  # for StepLR
        'weight_decay': 1e-4,
        'sub': 2 ** 0,
        'train_size': 1000,
        'test_size': 100,
        'degree': 40,
        'width': 50,

        'idctn' : functools.partial(ch.Wrapper, [ch.idct]),
        'dctn' : functools.partial(ch.Wrapper, [ch.dct]),
    }
    def load_paras(glob, para_list):
        for key, value in para_list.items():
            glob[key] = value

    load_paras(glob, default_para)
    load_paras(glob, para_list)

para_neumann = { # for experiment 1
    'data_PATH' : './data/burgers_neumann.mat',
    'sub_list': [2**4, 2**2, 2**0],
    'sub': 2**4,

    'x2phi' : functools.partial(ch.Wrapper, [ch.dct, ch.cmp_neumann]),
    'phi2x' : functools.partial(ch.Wrapper, [ch.icmp_neumann, ch.idct]),
}

para_robin = { # for experiment 2, with batch size 20
    'data_PATH' : './data/heat_robin1k.mat',
    'sub_list': [2**5, 2**3, 2**1],
    'sub': 2**5,
    'train_size': 1000,
    'test_size' : 1000,

    'x2phi': functools.partial(ch.Wrapper, [ch.dct, ch.cmp_robin]),
    'phi2x': functools.partial(ch.Wrapper, [ch.icmp_robin, ch.idct]),
}

para_robin_bs4 = { # for experiment 2, with batch size 4
    **para_robin,
    'batch_size': 4,
}

para_robin_bigdata = { # for experiment 2 against 2nd-order FDM
    'data_PATH' : './data/heat_robinN256.mat',
    'sub_list': [2**0],
    'sub': 2**0,
    'epochs' : 6000,
    'train_size': 100000,
    'test_size' : 1000,
    'learning_rate' : 0.001 * 0,

    'x2phi': functools.partial(ch.Wrapper, [ch.dct, ch.cmp_robin]),
    'phi2x': functools.partial(ch.Wrapper, [ch.icmp_robin, ch.idct]),
}

para_dirichlet = { # for experiment 3
    'data_PATH' : './data/heat_inho.mat',
    'sub_list': [2**5, 2**3, 2**1],
    'sub': 2**5,
    'train_size': 1000,
    'test_size' : 100,

    'x2phi': functools.partial(ch.Wrapper, [ch.dct, ch.cmp]),
    'phi2x': functools.partial(ch.Wrapper, [ch.icmp, ch.idct]),
}

para_burgers2d = { # for experiment 4
    'data_PATH' : './data/burgers2d.mat',
    'sub_list': [2**2, 2**1, 2**0],
    'sub': 2**2,
    'train_size': 1000,
    'test_size' : 100,
    'degree' : 16,
    'width' : 24,

    'x2phi' : functools.partial(ch.Wrapper, [ch.dct, ch.cmp_neumann]),
    'phi2x' : functools.partial(ch.Wrapper, [ch.icmp_neumann, ch.idct]),
}

if __name__ == "__main__":
    print('This is an a supplementary program for users to reproduce the results in the paper.')
    print('For example, users can insert the following code before printing the parameters:')
    print()
    print('from experiment_parameters import *')
    print('load_experiment_para(globals(), para_burgers_neumann)')
    print('print(sub_list)')
    print()
    print('However, it is NOT recommended because the code is too ugly and we have not fully check them.')

    # from experiment_parameters import *
    # load_experiment_para(globals(), para_burgers_neumann)
    # print(sub_list)
