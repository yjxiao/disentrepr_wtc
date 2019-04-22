from disent.criterions import NegativeELBOLoss
from . import BaseTask, register_task


@register_task('vae')
class VAETask(BaseTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--datadir', default='data/', type=str,
                            help='data directory')
        parser.add_argument('--dataset', default='dsprites', type=str,
                            help='dataset name to load')
        parser.add_argument('--kld-weight', default=1, type=float,
                            help='weight to the kld term')
        
    def build_criterion(self, args):
        return NegativeELBOLoss(args)
    
    def build_model(self, args):
        from disent import models
        return models.build_model(args)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # forward pass and handles kld_weight
        model.train()
        (rec, kld), batch_size, logging_output = criterion(model, sample)
        loss = rec + optimizer.get_hparam('kld_weight') * kld
        logging_output['loss'] = loss.item()
        if ignore_grad:
            loss *= 0
        return loss, batch_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            (rec, kld), batch_size, logging_output = criterion(model, sample)
        loss = rec + self.args.kld_weight * kld
        logging_output['loss'] = loss.item()
        return loss, batch_size, logging_output
