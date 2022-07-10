import torch
import numpy as np


class train():

    def __init__(self) -> None:
        pass

    def evaluate_accuracy(data_iter, net, device=None):
        if device is None and isinstance(net, torch.nn.Module):
            # 如果没指定device就使用net的device
            device = list(net.parameters())[0].device
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X, y in data_iter:
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X).argmax(dim=1) ==
                            y).float().sum().cpu().item()
                net.train()  # 改回训练模式
                n += y.shape[0]
        return acc_sum / n

    def single_process(x):
        x = np.array(x)
        return torch.tensor(x, dtype=torch.float32).reshape(1, 1, x.shape[0], x.shape[1])
