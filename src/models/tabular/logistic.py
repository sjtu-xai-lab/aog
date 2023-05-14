import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_logistic(nn.Module):

    def __init__(self, in_dim, hidd_dim, out_dim, n_layer):
        super(MLP_logistic, self).__init__()

        assert out_dim == 1

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")

        self.layers = self._make_layers(in_dim, hidd_dim, out_dim, n_layer)

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer):
        layers = [nn.Linear(in_dim, hidd_dim), nn.ReLU()]
        for _ in range(n_layer - 2):
            layers.extend([nn.Linear(hidd_dim, hidd_dim), nn.ReLU()])
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(hidd_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        return self.layers(x)

# ===========================
#   wrapper
# ===========================
def mlp2_logistic(in_dim, hidd_dim, out_dim):
    return MLP_logistic(in_dim, hidd_dim, out_dim, n_layer=2)



if __name__ == '__main__':
    x = torch.rand(1000,10)
    net = mlp2_logistic(in_dim=10, hidd_dim=100, out_dim=1)
    print(net)
    print(net(x).shape)