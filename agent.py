import torch
from rbfn import RBFNetwork

class DistributedAgent:
    def __init__(self, feature_dim, num_centers, alpha):
        self.model = RBFNetwork(feature_dim, num_centers, output_dim=8)
        self.alpha = alpha

    def compute_loss(self, features, targets):
        pred = self.model(features)
        return torch.mean((pred - targets) ** 2)

    def get_parameters(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()])

    def set_parameters(self, flat_param):
        idx = 0
        for p in self.model.parameters():
            num = p.numel()
            p.data.copy_(flat_param[idx:idx + num].view_as(p))
            idx += num
