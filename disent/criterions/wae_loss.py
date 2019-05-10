import torch
from torch.nn.modules.loss import _Loss

from .utils import get_gradient_penalty, shuffle_code


class WAELoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    @property
    def loss_components(self):
        return ['rec', 'div', 'critic_loss', 'grad_penalty']
    
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

        # W1 distance
        z_prior = outputs['prior'].rsample()
        f_z_pst = model['adversarial'](z)
        f_z_prr = model['adversarial'](z_prior)
        div = (f_z_pst.sum() - f_z_prr.sum()) / batch_size
        logging_output['div'] = div.item()
        
        # Adv loss
        if model.training:
            gp = get_gradient_penalty(z, z_prior, model['adversarial'])
            logging_output['grad_penalty'] = gp.item()
        else:
            gp = 0
            logging_output['grad_penalty'] = 0
        adv_loss = - div
        logging_output['critic_loss'] = adv_loss.item()
        

        return (rec, div, adv_loss, gp), batch_size, logging_output
