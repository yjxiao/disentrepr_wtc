import torch
from torch.nn.modules.loss import _Loss

from .utils import get_gradient_penalty, shuffle_code


class WTCWAELoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    @property
    def loss_components(self):
        return ['rec', 'wtc', 'dim_div', 'critic1', 'critic2', 'gp1', 'gp2']
    
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

        # WTC: W1 distance between q(z) and \prod_i(q(z_i))
        z_shuffled = shuffle_code(z)
        f_z_jnt = model['adv1'](z)
        f_z_mrg = model['adv1'](z_shuffled)
        wtc = (f_z_jnt.sum() - f_z_mrg.sum()) / batch_size
        logging_output['wtc'] = wtc.item()

        # Dimwise div: W1 distance between \prod_i(q(z_i)) and p(z)
        z_prior = outputs['prior'].rsample()
        g_z_mrg = model['adv2'](z_shuffled)
        g_z_prr = model['adv2'](z_prior)
        div = (g_z_mrg.sum() - g_z_prr.sum()) / batch_size
        logging_output['dim_div'] = div.item()
        
        # Adv loss
        if model.training:
            gp1 = get_gradient_penalty(z, z_shuffled, model['adv1'])
            gp2 = get_gradient_penalty(z_shuffled, z_prior, model['adv2'])
            logging_output['gp1'] = gp1.item()
            logging_output['gp2'] = gp2.item()
        else:
            gp1 = 0
            gp2 = 0
            logging_output['gp1'] = 0
            logging_output['gp2'] = 0

        critic1 = - wtc
        critic2 = - div
        logging_output['critic1'] = critic1.item()
        logging_output['critic2'] = critic2.item()        

        return (rec, wtc, div, critic1, critic2, gp1, gp2), batch_size, logging_output
