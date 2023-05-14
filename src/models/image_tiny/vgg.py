"""
This code is adapted from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py

    vgg in pytorch
    [1] Karen Simonyan, Andrew Zisserman
        Very Deep Convolutional Networks for Large-Scale Image Recognition.
        https://arxiv.org/abs/1409.1556v6
"""


import torch
import torch.nn as nn

__all__ = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

    def set_store_activation_rate(self):
        from .tools import AverageMeter

        def get_hook(name):
            def store_act_rate(m, i, o):
                self.activation_rate[name].update((o > 0).float().mean().item())
                print(o.shape)
            return store_act_rate

        if len(self.features) == 29:  # VGG-11
            self.activation_rate = {
                "conv_11": AverageMeter(), "conv_21": AverageMeter(), "conv_32": AverageMeter(),
                "conv_42": AverageMeter(), "conv_52": AverageMeter()
            }
            self.features[2].register_forward_hook(get_hook("conv_11"))
            self.features[6].register_forward_hook(get_hook("conv_21"))
            self.features[13].register_forward_hook(get_hook("conv_32"))
            self.features[20].register_forward_hook(get_hook("conv_42"))
            self.features[27].register_forward_hook(get_hook("conv_52"))
        elif len(self.features) == 35:  # VGG-13
            self.activation_rate = {
                "conv_12": AverageMeter(), "conv_22": AverageMeter(), "conv_32": AverageMeter(),
                "conv_42": AverageMeter(), "conv_52": AverageMeter()
            }
            self.features[5].register_forward_hook(get_hook("conv_12"))
            self.features[12].register_forward_hook(get_hook("conv_22"))
            self.features[19].register_forward_hook(get_hook("conv_32"))
            self.features[26].register_forward_hook(get_hook("conv_42"))
            self.features[33].register_forward_hook(get_hook("conv_52"))
        elif len(self.features) == 44:  # VGG-16
            self.activation_rate = {
                "conv_12": AverageMeter(), "conv_22": AverageMeter(), "conv_33": AverageMeter(),
                "conv_43": AverageMeter(), "conv_53": AverageMeter()
            }
            self.features[5].register_forward_hook(get_hook("conv_12"))
            self.features[12].register_forward_hook(get_hook("conv_22"))
            self.features[22].register_forward_hook(get_hook("conv_33"))
            self.features[33].register_forward_hook(get_hook("conv_43"))
            self.features[42].register_forward_hook(get_hook("conv_53"))
        elif len(self.features) == 53:  # VGG-19
            self.activation_rate = {
                "conv_12": AverageMeter(), "conv_22": AverageMeter(), "conv_34": AverageMeter(),
                "conv_44": AverageMeter(), "conv_54": AverageMeter()
            }
            self.features[5].register_forward_hook(get_hook("conv_12"))
            self.features[12].register_forward_hook(get_hook("conv_22"))
            self.features[25].register_forward_hook(get_hook("conv_34"))
            self.features[38].register_forward_hook(get_hook("conv_44"))
            self.features[51].register_forward_hook(get_hook("conv_54"))
        else:
            raise NotImplementedError


def make_layers(cfg, input_channel=3, batch_norm=False):
    layers = []

    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def vgg11_bn(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['A'], input_channel, batch_norm=True), num_classes)

def vgg13_bn(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['B'], input_channel, batch_norm=True), num_classes)

def vgg16_bn(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['D'], input_channel, batch_norm=True), num_classes)

def vgg19_bn(input_channel=3, num_classes=10):
    return VGG(make_layers(cfg['E'], input_channel, batch_norm=True), num_classes)


if __name__ == '__main__':
    print(torch.__version__)
    # net = vgg11_bn(input_channel=1, num_classes=10)
    # net = vgg13_bn(input_channel=1, num_classes=10)
    net = vgg16_bn(input_channel=1, num_classes=10)
    # net = vgg19_bn(input_channel=1, num_classes=10)
    x = torch.randn(1, 1, 32, 32)
    print(net(x).shape)
    # print(net.features)
