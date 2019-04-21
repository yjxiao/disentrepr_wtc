from torch.distributions import Normal
from torch.distributions import LogisticNormal
from torch.distributions import Dirichlet

DISTRIBUTION_REGISTRY = {
    'normal': Normal,
    'logistic_normal': LogisticNormal,
    'dirichlet': Dirichlet,
}

from .stochastic_module import StochasticModule


__all__ = [
    'StochasticModule',
]
