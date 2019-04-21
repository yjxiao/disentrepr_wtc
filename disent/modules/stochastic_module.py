import torch
import torch.nn as nn
from torch.distributions import constraints

from . import DISTRIBUTION_REGISTRY


class StochasticModule(nn.Module):
    def __init__(self, size, distribution='normal', learnable=False):
        super().__init__()
        self.size = size
        if distribution not in DISTRIBUTION_REGISTRY:
            raise ValueError("Unsupported distribution: " + distribution)
        self.Dist = DISTRIBUTION_REGISTRY[distribution]
        nargs = len(self.Dist.arg_constraints)
        if distribution == 'logistic_normal':
            size -= 1
        self.params = nn.ParameterList(
            [nn.Parameter(
                torch.zeros(size), requires_grad=learnable)
             for _ in range(nargs)])

    def forward(self, inputs):
        # notice the first dimension of inputs must be the batch size
        batch_size = inputs.detach().size(0)
        kwargs = {}
        for (arg, constr), param in zip(self.Dist.arg_constraints.items(), self.params):
            if constr is constraints.positive:
                param = param.exp()
            kwargs[arg] = param.unsqueeze(0).expand(batch_size, -1)
        return self.Dist(**kwargs)
