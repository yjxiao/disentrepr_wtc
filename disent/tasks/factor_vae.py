import torch.nn as nn

from disent.criterions import FactorVAELoss
from . import BaseTask, register_task


@register_task('factor_vae')
class FactorVAETask(BaseTask):

    def build_criterion(self, args):
        return FactorVAELoss(args)
    
    def build_model(self, args):
        from disent import models
        return nn.ModuleDict({
            'main': models.build_model(args),
            'adversarial': models.build_adversarial(args)
        })

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, kld, tc, adv_loss), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.kld_weight * kld + optimizer.beta * tc
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
            (rec, kld, tc, adv_loss), batch_size, logging_output = criterion(model, sample)
        loss = rec + kld + args.beta * tc
        losses = {
            'main': loss,
            'adversarial': adv_loss,
        }
        return losses, batch_size, logging_output
