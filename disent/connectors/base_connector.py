import torch.nn as nn


class BaseConnector(nn.Module):
    """Base class for connectors."""
    def __init__(self, output_size):
        super().__init__()
        self._output_size = output_size
        
    def forward(self, inputs):
        raise NotImplementedError
