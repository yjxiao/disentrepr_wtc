import torch
from torch.nn.modules.loss import _Loss
from torch.distributions import kl_divergence


class NegativeELBOLoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, model, sample):
        outputs = model(sample)
        batch_size = sample['batch_size']
        
        logging_output = {}
        logging_output['batch_size'] = batch_size
        
        kld = torch.sum(
            kl_divergence(outputs['posterior'], outputs['prior'])) / batch_size
        logging_output['kld'] = kld.item()

        x_recon = outputs['x']
        # p(x|z)
        rec = torch.sum(- x_recon.log_prob(sample['image'])) / batch_size
        logging_output['rec'] = rec.item()

        return (rec, kld), batch_size, logging_output
