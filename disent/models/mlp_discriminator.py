import torch.nn as nn

from disent.connectors import MLPConnector
from disent.utils import getattr_with_default
from . import BaseModel, register_model


@register_model('mlp_discriminator')
class MLPDiscriminator(BaseModel):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    @staticmethod
    def add_args(parser):
        parser.add_argument('--discriminator-input-size', type=int,
                            help='input size to the discriminator. default to code size')
        parser.add_argument('--discriminator-layers', type=str, metavar='EXPR',
                            help='dense layers (hidden_size, ...)')
        parser.add_argument('--discriminator-nonlinearity', type=str, default='leaky_relu',
                            help='activation function used in the mlp')
        
    @classmethod
    def build_model(cls, args):
        base_architecture(args)

        layers = []
        input_size = args.discriminator_input_size
        for output_size in args.discriminator_layers:
            layers.append(
                MLPConnector(
                    input_size, output_size, args.discriminator_nonlinearity)
            )
            input_size = output_size

        layers.append(nn.Linear(input_size, 2))
        layers.append(nn.LogSoftmax(dim=1))
        return cls(nn.Sequential(*layers))

    def forward(self, inputs):
        # inputs size: (batch_size, input_size)
        # return logits
        return self.layers(inputs)


def base_architecture(args):
    args.discriminator_input_size = getattr_with_default(
        args, 'discriminator_input_size', args.code_size)
    d_layers = getattr_with_default(
        args, 'discriminator_layers', '(1000,) * 6')
    if isinstance(d_layers, str):
        d_layers = eval(d_layers)
    args.discriminator_layers = d_layers
    args.discriminator_nonlinearity = getattr_with_default(
        args, 'discriminator_nonlinearity', 'leaky_relu')
