import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    """
    Max-Feature-Map activation used in Light CNN.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        first, second = torch.chunk(x, chunks=2, dim=self.dim)
        return torch.maximum(first, second)


class MFMConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, *args, **kwargs)
        self.mfm = MaxFeatureMap(dim=1)

    def forward(self, x):
        return self.mfm(self.conv(x))


class MFMLinear(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2, *args, **kwargs)
        self.mfm = MaxFeatureMap(dim=1)

    def forward(self, x):
        return self.mfm(self.linear(x))
