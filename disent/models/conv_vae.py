import numpy as np
import torch
import torch.nn as nn

from disent.connectors import (
    BernoulliWrappingConnector, MLPConnector, StochasticConnector)
from disent.modules import StochasticModule
from . import BaseModel, register_model


@register_model('conv_vae')
class ConvVAE(BaseModel):
    def __init__(self, encoder, decoder, prior):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    @staticmethod
    def add_args(parser):
        parser.add_argument('--code-size', type=int, default=10,
                            help='latent code dimension')
        parser.add_argument('--conv-layers', type=str, metavar='EXPR',
                            help='convolution layers [(out_channels, kernel_size, stride, padding), ...]')
        parser.add_argument('--conv-output-size', type=str, metavar='EXPR',
                            help='output size from the last convolution layer (out_channels, height, width)')
        parser.add_argument('--dense-layers', type=str, metavar='EXPR',
                            help='dense layers (hidden_size, ...)')
        parser.add_argument('--batch-norm', action='store_true',
                            help='add batch norm layers after conv layers')
        parser.add_argument('--learnable-prior', action='store_true',
                            help='make the prior learnable')

    @classmethod
    def build_model(cls, args):
        # default values if not given
        base_architecture(args)
        
        encoder = ConvEncoder(
            args.code_size, args.in_channels,
            args.encoder_convs, args.encoder_denses,
            args.batch_norm)

        decoder = ConvDecoder(
            args.code_size, args.conv_output_size,
            args.decoder_convs, args.decoder_denses,
            args.batch_norm)

        prior = StochasticModule(args.code_size, 'normal', args.learnable_prior)
        return cls(encoder, decoder, prior)

    def forward(self, samples):
        images = samples['image']
        posterior = self.encoder(images)
        if self.training:
            z = posterior.rsample()
        else:
            z = posterior.mean
        prior = self.prior(z)
        x = self.decoder(z)
        return {
            'x': x;
            'posterior': posterior,
            'prior': prior,
            'z': z
        }


class ConvEncoder(nn.Module):
    """Convolutional encoder for VAE

    Args:
        in_channels (int): number of input channels
        code_size (int): latent code dimension
        convolutions (list): convolution layer structure. Layers are given as 
            (out_channels, kernel_size, stride, padding)
        fcs (list): dense layer structure. Each element represents number of neurons
            in the corresponding layer. The first element has to be consistent with 
            output from the convolution layers
    """
    
    def __init__(self, code_size=10, in_channels=1,
                 convolutions=((32, 4, 2, 1),) * 2 + ((64, 4, 2, 1),) * 2,
                 fcs=(1024, 256), batch_norm=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self._use_bn = batch_norm
        for conv in convolutions:
            out_channels, kernel_size, stride, padding = conv
            self.convs.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding))
            self.bns.append(
                nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        self.fcs = nn.ModuleList()
        in_feats = fcs[0]
        for out_feats in fcs[1:]:
            self.fcs.append(
                MLPConnector(in_feats, out_feats, 'relu'))
            in_feats = out_feats

        self.h2z = StochasticConnector(in_feats, code_size, 'normal')

    def forward(self, inputs):
        # size (batch_size, in_channels, height, weight)
        outputs = inputs
        for conv, bn in zip(self.convs, self.bns):
            outputs = conv(outputs)
            if self._use_bn:
                outputs = bn(outputs)
            outputs = torch.relu(outputs)
        outputs = outputs.view(inputs.size(0), -1)
        for fc in self.fcs:
            outputs = fc(outputs)

        dist = self.h2z(outputs)
        return dist


class ConvDecoder(nn.Module):
    """Convolutional decoder for VAE

    Args:
        in_sizes (list): input size to the first convolution layer (in_channels, height, weight)
        code_size (int): latent code dimension
        convolutions (list): convolution layer structure. Layers are given as 
            (out_channels, kernel_size, stride, padding)
        fcs (list): dense layer structure. Each element represents number of neurons
            in the corresponding layer. The first element has to be consistent with 
            output from the convolution layers
    """
    
    def __init__(self, code_size=10, in_sizes=(64, 4, 4),
                 convolutions=((64, 4, 2, 1),) + ((32, 4, 2, 1),) * 2 + ((1, 4, 2, 1),),
                 fcs=(256,), batch_norm=False):
        super().__init__()

        self.fcs = nn.ModuleList()
        in_feats = code_size
        for out_feats in fcs:
            self.fcs.append(MLPConnector(in_feats, out_feats, 'relu'))
            in_feats = out_feats
        self.fcs.append(
            MLPConnector(in_feats, np.prod(in_sizes), 'relu'))
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self._use_bn = batch_norm

        self._in_sizes = in_sizes
        in_channels = in_sizes[0]
        for i, conv in enumerate(convolutions):
            out_channels, kernel_size, stride, padding = conv
            self.convs.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding))
            if i < len(convolutions) - 1:
                self.bns.append(
                    nn.BatchNorm2d(out_channels))
                in_channels = out_channels
        self.bernoulli_wrapper = BernoulliWrappingConnector()

    def forward(self, inputs):
        outputs = inputs    # size (batch_size, code_size)
        for fc in self.fcs:
            outputs = fc(outputs)

        outputs.view(-1, *self._in_sizes)
        for conv, bn in zip(self.convs[:-1], self.bns):
            outputs = conv(outputs)
            if self._use_bn:
                outputs = bn(outputs)
            outputs = torch.relu(outputs)
        # no batch norm or activation on the final deconvolution layer
        outputs = self.convs[-1](outputs)
        return self.bernoulli_wrapper(outputs)


def base_architecture(args):
    conv_layers = eval(getattr(args, 'conv_layers', '((32, 4, 2, 1),) * 2 + ((64, 4, 2, 1),) * 2'))
    dense_layers = eval(getattr(args, 'dense_layers', '(1024, 256)'))
    out_size = eval(getattr(args, 'conv_output_size', '(64, 4, 4)'))
    
    args.encoder_convs = conv_layers
    args.encoder_denses = dense_layers
    args.conv_output_size = out_size

    # reverse encoder layers
    args.decoder_convs = reversed(conv_layers[:-1]) + ((args.in_channels,) + conv_layers[-1][1:])
    args.decoder_denses = reversed(dense_layers[1:])
