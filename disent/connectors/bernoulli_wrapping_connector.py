import torch.nn.functional as F
from torch.distributions import Bernoulli

from . import BaseConnector


class BernoulliWrappingConnector(BaseConnector):
    """Connector that wraps tensor with real values to a bernoulli distribution. """

    def forward(self, inputs):
        return Bernoulli(logits=inputs)
