import torch
import torch.nn as nn

from disent.connectors import MLPConnector, StochasticConnector
from disent.utils import getattr_with_default
from . import BaseModel, register_model


@register_model('mlp_regressor')
class MLPRegressor(BaseModel):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    @staticmethod
    def add_args(parser):
        parser.add_argument('--regressor-input-size', type=int,
                            help='input size to the regressor. default to code size')
        parser.add_argument('--regressor-layers', type=str, metavar='EXPR',
                            help='dense layers (hidden_size, ...)')
        parser.add_argument('--regressor-nonlinearity', type=str, default='relu',
                            help='activation function used in the mlp')
        parser.add_argument('--regressor-output-distribution', type=str,
                            help='output distribution. default to None (scalar output)')
        
    @classmethod
    def build_model(cls, args):
        base_architecture(args)

        layers = []
        input_size = args.regressor_input_size
        for output_size in args.regressor_layers:
            layers.append(
                MLPConnector(
                    input_size, output_size,
                    args.regressor_nonlinearity)
            )
            input_size = output_size

        if args.regressor_output_distribution is None:
            layers.append(
                MLPConnector(
                    input_size, 1, nonlinearity=None,
                    squeeze_output=True
                ))
        else:
            layers.append(
                StochasticConnector(
                    input_size, 1,
                    distribution=args.regressor_output_distribution,
                    squeeze_output=True
                ))
        return cls(nn.Sequential(*layers))

    def forward(self, inputs):
        return self.layers(inputs)


def base_architecture(args):
    args.regressor_input_size = getattr_with_default(
        args, 'regressor_input_size', args.code_size)
    r_layers = getattr_with_default(
        args, 'regressor_layers', '(256,) * 3')
    if isinstance(r_layers, str):
        r_layers = eval(r_layers)
    args.regressor_layers = r_layers
    args.regressor_nonlinearity = getattr_with_default(
        args, 'regressor_nonlinearity', 'relu')
