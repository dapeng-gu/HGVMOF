from typing import  Text, List

import torch.nn as nn

def make_mlp(layer_dims: List[int], act: Text = 'relu', batchnorm: bool = False, dropout: float = 0.0,
             activation_last: bool = False) -> nn.Sequential:
    layers = []
    n_layers = len(layer_dims)
    assert n_layers >= 2, "MLP should be at least 2 layers."
    last_layer = n_layers - 2
    for index in range(last_layer + 1):
        layers.append(nn.Linear(layer_dims[index], layer_dims[index + 1]))
        if index < last_layer:
            layers.append(get_activation(act))
        elif activation_last:
            layers.append(get_activation(act))
        if batchnorm:
            layers.append(nn.BatchNorm1d(layer_dims[index + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)

def get_activation(astr: Text) -> nn.modules.activation:
    activations = {'selu': nn.SELU, 'relu': nn.ReLU, 'prelu': nn.PReLU, 'leaky_relu': nn.LeakyReLU,
                   'softplus': nn.Softplus}
    return activations[astr]()