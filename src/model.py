import torch.nn as nn

def build_regression_model(layer_sizes):
    """
    Builds a flexible sequential neural network for regression using PyTorch.

    Args:
        layer_sizes (list): A list of integers defining the size of each layer.

    Returns:
        torch.nn.Sequential: The PyTorch model.
    """
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        layers.append(nn.ReLU())
    # Add the final layer without a non-linear activation for regression
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    
    return nn.Sequential(*layers)