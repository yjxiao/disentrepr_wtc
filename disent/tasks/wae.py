import torch
import torch.nn as nn

from disent.criterions import WAELoss
from disent.utils import eval_str_list
from . import BaseTask, register_task


@register_task('wae')
class WAETask(BaseTask):
    hparams = ('beta', 'lambda')

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', default='data/', type=str,
                            help='data directory')
        parser.add_argument('--dataset', default='dsprites', type=str,
                            help='dataset name to load')
        parser.add_argument('--adversarial-arch', metavar='ARCH',
                            type=str, default='mlp_regressor',
                            help='adversarial model architecture')
        parser.add_argument('--beta', default='1', type=eval_str_list,
                            help='weight to the kld term')
        parser.add_argument('--lambda', default='8', type=eval_str_list,
                            help='weight to the gradient penalty term')
        
    def build_criterion(self, args):
        return WAELoss(args)
    
    def build_model(self, args):
        from disent import models
        return nn.ModuleDict({
            'main': models.build_model(args),
            'adversarial': models.build_adversarial(args)
        })

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, div, critic, gpenalty), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.get_hparam('beta') * div
        logging_output['loss'] = loss.item()
        adv_loss = critic + optimizer.get_hparam('lambda') * gpenalty

        if ignore_grad:
            loss *= 0
            adv_loss *= 0

        losses = {
            'main': loss,
            'adversarial': adv_loss,
        }
        return losses, batch_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            (rec, div, critic, gpenalty), batch_size, logging_output = criterion(model, sample)
        loss = rec + self.args.beta[-1] * div
        logging_output['loss'] = loss.item()
        losses = {
            'main': loss,
            'adversarial': critic,    # ignore gradient penalty during evaluation
        }
        return loss, batch_size, logging_output
