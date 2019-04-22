from disent.criterions import TCVAELoss
from . import BaseTask, register_task


@register_task('tc_vae')
class TCVAETask(BaseTask):

    def build_criterion(self, args):
        return TCVAELoss(args)
    
    def build_model(self, args):
        from disent import models
        return models.build_model(args)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, kld, tc), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.kld_weight * kld + optimizer.beta * tc
        if ignore_grad:
            loss *= 0
        return loss, batch_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            (rec, kld, tc), batch_size, logging_output = criterion(model, sample)
        loss = rec + kld + args.beta * tc
        return loss, batch_size, logging_output
