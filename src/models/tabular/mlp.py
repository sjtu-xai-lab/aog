import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, hidd_dim, out_dim, n_layer):
        super(MLP, self).__init__()

        if n_layer < 2:
            raise Exception(f"Invalid #layer: {n_layer}.")

        self.layers = self._make_layers(in_dim, hidd_dim, out_dim, n_layer)

    def _make_layers(self, in_dim, hidd_dim, out_dim, n_layer):
        layers = [nn.Linear(in_dim, hidd_dim), nn.ReLU()]
        for _ in range(n_layer - 2):
            layers.extend([nn.Linear(hidd_dim, hidd_dim), nn.ReLU()])
        layers.append(nn.Linear(hidd_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        return self.layers(x)

    def get_feature(self, x, layer_id: int):
        '''
        :param x:
        :param layer_id: layer_id can be 0, 1, 2, 3 ....
        :return: output feautre after [layer_id] in nn.Sequential(...)
        '''
        x = x.reshape(x.shape[0], -1)
        return self.layers[:layer_id+1](x)

    def load_classifier(self, teacher, after: int):
        '''
        load the parameters in layer [after+1], [after+2], ...
        :param teacher:
        :param after:
        :return:
        '''
        params_to_update = teacher.layers[after+1:].state_dict()
        state_dict = self.layers.state_dict()
        state_dict.update(params_to_update)
        self.layers.load_state_dict(state_dict)

# ===========================
#   wrapper
# ===========================
def mlp5(in_dim, hidd_dim, out_dim):
    return MLP(in_dim, hidd_dim, out_dim, n_layer=5)



if __name__ == '__main__':
    x = torch.rand(1000,10)
    net = mlp5(in_dim=10, hidd_dim=100, out_dim=2)
    print(net)
    print(net(x).shape)
    print(net.get_feature(x, layer_id=6).shape)
    print(net.get_feature(x, layer_id=6).min())