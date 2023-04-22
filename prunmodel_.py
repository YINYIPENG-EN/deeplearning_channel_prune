from tools.utils import plot_weights, plot_3D_weights, count_params, prune
import torch
from mymodel import Model
import time
import argparse


def run(model):
    prune_, percent, save_model, plt, plt_3d, layer_name = opt.prune, opt.percent, opt.save, opt.plt, opt.plt_3d, opt.layer_name
    x = torch.randn(1, 3, 32, 32)
    t1 = time.time()
    _ = model(x)
    t2 = time.time()
    print("inference time: {:.2f} second".format(t2-t1))
    total_param = count_params(model)
    if save_model:  torch.save(model, "model.pth")
    print(f'\033[5;33m model: {model}\033[0m')
    print("Number of parameter: %.2fM" % (total_param / 1e6))
    if prune_:

        prune_model = prune(model, percent)
        print(f'\033[1;36m pruned model: {prune_model}\033[0m')
        total_prune_param = count_params(prune_model)

        print("Number of pruned model parameter: %.2fM" % (total_prune_param / 1e6))
        if save_model: torch.save(prune_model, "pruned.pth")
        t1 = time.time()
        out = prune_model(x)
        t2 = time.time()
        print("pruned model inference time: {:.2f} second".format(t2 - t1))
    if plt:
        plot_weights(model, layer_name)
    elif plt_3d:
        plot_3D_weights(model, layer_name)


if __name__ == '__main__':
    parse = argparse.ArgumentParser('channel prune')
    parse.add_argument('--prune', action='store_true', default=False, help='prune model')
    parse.add_argument('--percent', type=float, default=0.5, help='prune percent')
    parse.add_argument('--save', action='store_true', default=False, help='save model')
    parse.add_argument('--plt', action='store_true', default=False, help='plot 2D conv weight')
    parse.add_argument('--plt_3d', action='store_true', default=False, help='plot 3D conv weight')
    parse.add_argument('--layer_name', default=['conv1.weight'], help='plot conv name')
    opt = parse.parse_args()
    print(opt)

    model = Model(3)  # 自己的模型
    run(model)


