import torch.nn as nn


class BaseConnector(nn.Module):
    """Base class for connectors."""
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        raise NotImplementedError
