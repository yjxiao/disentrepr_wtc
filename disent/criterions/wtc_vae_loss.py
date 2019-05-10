import torch
from torch.nn.modules.loss import _Loss
from torch.distributions import kl_divergence

from .utils import get_gradient_penalty, shuffle_code


class WTCVAELoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    @property
    def loss_components(self):
        return ['rec', 'kld', 'wtc', 'critic_loss', 'gradient_penalty']
    
    def forward(self, model, sample):
        outputs = model['main'](sample)
        batch_size = sample['batch_size']
        z = outputs['z']
        x_rec = outputs['x']
        logging_output = {}
        logging_output['batch_size'] = batch_size

        # Rec
        # p(x|z)
        rec = torch.sum(- x_rec.log_prob(sample['image'])) / batch_size
        logging_output['rec'] = rec.item()

        # KLD
        kld = torch.sum(
            kl_divergence(outputs['posterior'], outputs['prior'])) / batch_size
        logging_output['kld'] = kld.item()
        
        # WTC
        shuffled_z = shuffle_code(z)
        f_z_jnt = model['adversarial'](z)
        f_z_mrg = model['adversarial'](shuffled_z)
        wtc = (f_z_jnt.sum() - f_z_mrg.sum()) / batch_size
        logging_output['wtc'] = wtc.item()
        
        # Adv loss
        if model.training:
            gp = get_gradient_penalty(z, shuffled_z, model['adversarial'])
            logging_output['gradient_penalty'] = gp.item()
        else:
            gp = 0
            logging_output['gradient_penalty'] = 0
        adv_loss = - wtc
        logging_output['critic_loss'] = adv_loss.item()

        return (rec, kld, wtc, adv_loss, gp), batch_size, logging_output
