import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

def count_params(module):
    return sum([p.numel() for p in module.parameters()])

def prune(model, percentage):
    # 计算每个通道的L1-norm并排序
    importance = {}
    prune_model = model
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            logger.info("module: ", name)
            # torch.norm用于计算张量的范数,可以计算每个通道上的L1范数 conv.weight.data shape [out_channels,in_channels, k,k]
            importance[name] = torch.norm(module.weight.data, 1, dim=(1, 2, 3))
            # 对通道进行排序,返回索引
            sorted_channels = np.argsort(np.concatenate([x.cpu().numpy().flatten() for x in importance[name]]))
            # logger.info(f"{name} layer channel sorting results {sorted_channels}")
            # 要剪掉的通道数量
            num_channels_to_prune = int(len(sorted_channels) * percentage)
            logger.info(
                f"The number of channels that need to be cut off in the {name} layer is {num_channels_to_prune}")
            logger.info(f"{name} layer pruning channel index is {sorted_channels[:num_channels_to_prune]}")

            new_module = nn.Conv2d(in_channels=3 if module.in_channels == 3 else in_channels,  # *
                                   out_channels=module.out_channels - num_channels_to_prune,
                                   kernel_size=module.kernel_size,
                                   stride=module.stride,
                                   padding=module.padding,
                                   dilation=module.dilation,
                                   groups=module.groups,
                                   bias=(module.bias is not None)
                                   ).to(next(model.parameters()).device)
            in_channels = new_module.out_channels  # 因为前一层的输出通道会影响下一层的输入通道
            # 重新分配权重 权重的shape[out_channels, in_channels, k, k]
            c2, c1, _, _ = new_module.weight.data.shape
            new_module.weight.data[...] = module.weight.data[num_channels_to_prune:, :c1, ...]
            if module.bias is not None:
                new_module.bias.data[...] = module.bias.data[num_channels_to_prune:, :c1, ...]
            # 用新卷积替换旧卷积
            setattr(prune_model, f"{name}", new_module)
    return prune_model

def plot_weights(model, layer_name):
    for name, param in model.named_parameters():
        if name in layer_name:
            plt.figure()
            plt.title(name)
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            values = param.cpu().data.numpy().flatten()
            mean = np.mean(values)
            std = np.std(values)
            plt.text(0.95, 0.95, 'Mean:{:.2f}\nStd: {:.2f}'.format(mean, std), transform=plt.gca().transAxes,
                     ha='right',
                     va='top')
            sns.histplot(values, kde=False, bins=50)
            plt.show()

def plot_3D_weights(model, layer_name):
    for name, param in model.named_parameters():
        print(name)
        if name in layer_name:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            values = param.cpu().data.numpy().flatten()
            # x对应输出通道，y对应输入通道
            x, y, z = np.indices((param.shape[0], param.shape[1], param.shape[2] * param.shape[3]))
            ax.scatter(x, y, z, c=values, cmap='jet')
            fig.colorbar(ax.get_children()[0], ax=ax)
            ax.set_xlabel('out_channels')
            ax.set_ylabel('in_channels')
            ax.set_zlabel('values')
            plt.title(name + ' weights distribution')
            plt.show()
