import torch.nn.functional as F
from torch.distributions import Bernoulli

from . import BaseConnector


class BernoulliWrappingConnector(BaseConnector):
    """Connector that wraps tensor with real values to a bernoulli distribution. """

    def forward(self, inputs):
        # real values are converted to probabilities using sigmoid function
        logits = F.logsigmoid(inputs)
        return Bernoulli(logits=logits)
