# NOTE: this model does not work
import torch
import torch.nn as nn

from disent.criterions import DualTCVAELoss
from disent.utils import eval_str_list
from . import BaseTask, register_task


@register_task('dualtc_vae')
class DualTCVAETask(BaseTask):
    hparams = ('kld_weight', 'beta')
    
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
                            help='extra weight to the tc component')
    
    def build_criterion(self, args):
        return DualTCVAELoss(args)
    
    def build_model(self, args):
        from disent import models
        return nn.ModuleDict({
            'main': models.build_model(args),
            'adversarial': models.build_adversarial(args)
        })

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, kld, dualtc, adv_loss), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.get_hparam('kld_weight') * kld + \
               optimizer.get_hparam('beta') * dualtc
        logging_output['loss'] = loss.item()
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
            (rec, kld, dualtc, adv_loss), batch_size, logging_output = criterion(model, sample)
        loss = rec + kld + self.args.beta[-1] * dualtc
        logging_output['loss'] = loss.item()
        
        losses = {
            'main': loss,
            'adversarial': adv_loss,
        }
        return losses, batch_size, logging_output
