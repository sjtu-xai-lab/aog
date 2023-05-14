import time
import torch
from torch import nn, optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        if in_dim == out_dim:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        if self.shortcut is None:
            out = out + x
        else:
            out = out + self.shortcut(x)
        out = F.relu(out)
        return out

    def forward_wo_relu(self, x):
        out = self.linear(x)
        if self.shortcut is None:
            out = out + x
        else:
            out = out + self.shortcut(x)
        return out


class ResMLP(nn.Module):
    def __init__(self, in_dim, hidd_dim, out_dim, n_layer):
        super(ResMLP, self).__init__()

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")

        self.layers = self._make_layers(in_dim, hidd_dim, out_dim, n_layer)

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer):
        layers = [nn.Linear(in_dim, hidd_dim), nn.ReLU()]
        for _ in range(n_layer - 2):
            layers.append(ResidualBlock(hidd_dim, hidd_dim))
        # layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(hidd_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)

    def get_feature(self, x, layer_id: int):
        x = x.reshape(x.shape[0], -1)
        x = self.layers[:layer_id](x)
        return self.layers[layer_id].forward_wo_relu(x)

    def load_classifier(self, teacher, after: int):
        params_to_update = teacher.layers[after + 1:].state_dict()
        state_dict = self.layers.state_dict()
        state_dict.update(params_to_update)
        self.layers.load_state_dict(state_dict)


# ===========================
#   wrapper
# ===========================
def resmlp5(in_dim, hidd_dim, out_dim):
    return ResMLP(in_dim, hidd_dim, out_dim, n_layer=5)


if __name__ == '__main__':
    x = torch.rand(1000,10)
    net = resmlp5(in_dim=10, hidd_dim=100, out_dim=2)
    print(net)
    print(net(x).shape)
    print(net.get_feature(x, 4).shape)
    print(net.get_feature(x, 4).min())