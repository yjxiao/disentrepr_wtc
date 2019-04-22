import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseConnector


class MLPConnector(BaseConnector):
    """Connector that uses an MLP to transform shape. (It is an MLP)"""
    
    def __init__(self, input_size, output_size, nonlinearity='tanh'):
        super().__init__(output_size)
        if nonlinearity in ['tanh', 'relu']:
            self.activation_func = getattr(torch, nonlinearity)
        elif nonlinearity in ['softplus', 'leaky_relu']:
            self.activation_func = getattr(F, nonlinearity)
        elif nonlinearity == 'none' or nonlinearity is None:
            self.activation_func = lambda x: x
        else:
            raise ValueError("Unsupported nonlinearity: " + nonlinearity)
        
        if type(output_size) is int:
            output_size = [output_size]
        self.linears = nn.ModuleList([nn.Linear(input_size, size) for size in output_size])
        
    def forward(self, inputs):
        results = tuple(
            self.activation_func(linear(inputs)) for linear in self.linears)
        if type(self._output_size) is int:
            return results[0]
        else:
            return results


class LinearConnector(MLPConnector):
    """Linear transformation."""
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, nonlinearity='none')
