import torch
import torch.nn as nn

from disent.criterions import WTCVAELoss
from disent.utils import eval_str_list
from . import BaseTask, register_task


@register_task('wtc_vae')
class WTCVAETask(BaseTask):
    """Wasserstein Total Correlation VAE. """
    hparams = ('beta', 'gamma')

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', default='data/', type=str,
                            help='data directory')
        parser.add_argument('--dataset', default='dsprites', type=str,
                            help='dataset name to load')
        parser.add_argument('--mmd-kernel', default='imq', type=str,
                            choices=['imq', 'rbf'],
                            help='type of kernel to use for mmd')
        parser.add_argument('--beta', default='1', type=eval_str_list,
                            help='weight to the divergence term')
        parser.add_argument('--gamma', default='1', type=eval_str_list,
                            help='weight to the wtc term')
        
    def build_criterion(self, args):
        return WTCVAELoss(args)
        
    def build_model(self, args):
        from disent import models
        return models.build_model(args)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, kld, wtc), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.get_hparam('beta') * kld + \
               optimizer.get_hparam('gamma') * wtc
        logging_output['loss'] = loss.item()
        if ignore_grad:
            loss *= 0
        return loss, batch_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            (rec, kld, wtc), batch_size, logging_output = criterion(model, sample)
        loss = rec + self.args.beta[-1] * kld + self.args.gamma[-1] * wtc
        logging_output['loss'] = loss.item()
        return loss, batch_size, logging_output
