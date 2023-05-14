'''
  Reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def set_store_activation_rate(self):

        from .tools import AverageMeter

        self.activation_rate = {"conv1": AverageMeter(), "conv2": AverageMeter()}

        def get_hook(name):
            def store_act_rate(m, i, o):
                self.activation_rate[name].update((o > 0).float().mean().item())
            return store_act_rate

        self.conv1.register_forward_hook(get_hook("conv1"))
        self.conv2.register_forward_hook(get_hook("conv2"))


def lenet(input_channel=1, num_classes=10):
    return LeNet(input_channel, num_classes)


if __name__ == '__main__':
    net = lenet()
    x = torch.randn(64, 1, 32, 32)
    print(net(x).shape)