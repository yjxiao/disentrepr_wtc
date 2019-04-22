from disent.criterions import NegativeELBOLoss
from . import BaseTask, register_task


@register_task('vae')
class VAETask(BaseTask):

    def build_criterion(self, args):
        return NegativeELBOLoss(args)
    
    def build_model(self, args):
        from disent import models
        return models.build_model(args)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, kld), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.kld_weight * kld
        if ignore_grad:
            loss *= 0
        return loss, batch_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            (rec, kld), batch_size, logging_output = criterion(model, sample)
        loss = rec + kld
        return loss, batch_size, logging_output
