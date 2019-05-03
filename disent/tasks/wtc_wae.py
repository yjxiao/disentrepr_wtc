import torch
import torch.nn as nn

from disent.criterions import WTCWAELoss
from disent.utils import eval_str_list
from . import BaseTask, register_task


@register_task('wtc_wae')
class WTCWAETask(BaseTask):
    hparams = ('beta', 'gamma', 'lambda')

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
                            help='weight to the wtc term')
        parser.add_argument('--gamma', default='1', type=eval_str_list,
                            help='weight to the dimwise distance term')
        parser.add_argument('--lambda', default='8', type=eval_str_list,
                            help='weight to the gradient penalty term')
        
    def build_criterion(self, args):
        return WTCWAELoss(args)
    
    def build_model(self, args):
        from disent import models
        return nn.ModuleDict({
            'main': models.build_model(args),
            'adv1': models.build_adversarial(args),
            'adv2': models.build_adversarial(args),
        })

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, wtc, div, critic1, critic2, gp1, gp2), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.get_hparam('beta') * wtc +\
               optimizer.get_hparam('gamma') * div
        logging_output['loss'] = loss.item()
        
        adv1_loss = critic1 + optimizer.get_hparam('lambda') * gp1
        adv2_loss = critic2 + optimizer.get_hparam('lambda') * gp2

        if ignore_grad:
            loss *= 0
            adv1_loss *= 0
            adv2_loss *= 0            

        losses = {
            'main': loss,
            'adv1': adv1_loss,
            'adv2': adv2_loss,
        }
        return losses, batch_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            (rec, wtc, div, critic1, critic2, gp1, gp2), batch_size, logging_output = criterion(model, sample)
        loss = rec + self.args.beta[-1] * wtc + self.args.gamma[-1] * div
        logging_output['loss'] = loss.item()
        losses = {
            'main': loss,
            'adv1': critic1,    # ignore gradient penalty during evaluation
            'adv2': critic2,    # ignore gradient penalty during evaluation            
        }
        return losses, batch_size, logging_output
