import torch
import torch.nn as nn
from torch.distributions import constraints

from disent.modules import DISTRIBUTION_REGISTRY
from . import BaseConnector


class StochasticConnector(BaseConnector):
    """Connector to produce a stochastic unit."""
    def __init__(self, input_size, output_size, distribution='normal'):
        super().__init__(output_size)
        if type(self._output_size) is not int:
            raise ValueError("Unsupported output size. Should be int.")
        if distribution not in DISTRIBUTION_REGISTRY:
            raise ValueError("Unsupported distribution: " + distribution)
        self.Dist = DISTRIBUTION_REGISTRY[distribution]
        nargs = len(self.Dist.arg_constraints)
        if distribution == 'logistic_normal':  # special treatment for logistic normal
            output_size -= 1
        self.linears = nn.ModuleList(
            [nn.Linear(input_size, output_size) for _ in range(nargs)])

    def forward(self, inputs):
        params = [linear(inputs) for linear in self.linears]
        kwargs = {}
        for (arg, constr), value in zip(self.Dist.arg_constraints.items(), params):
            if constr is constraints.positive:
                value = value.exp()    # ensure positivity
            kwargs[arg] = value
        return self.Dist(**kwargs)
