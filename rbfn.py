import torch
import torch.nn as nn


class RBFNetwork(nn.Module):
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFNetwork, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.beta = nn.Parameter(torch.ones(num_centers))
        self.linear = nn.Linear(num_centers, output_dim, bias=False)

    def _rbf(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        return torch.exp(-self.beta * torch.sum(diff ** 2, dim=2))

    def forward(self, x):
        phi = self._rbf(x)
        return self.linear(phi)
