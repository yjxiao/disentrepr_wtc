import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import kl_divergence


class TCVAELoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def loss_components(self):
        return ['rec', 'kld', 'tc']
    
    def forward(self, model, sample):
        outputs = model(sample)
        batch_size = sample['batch_size']
        posterior = outputs['posterior']
        
        logging_output = {}
        logging_output['batch_size'] = batch_size
        
        kld = torch.sum(kl_divergence(posterior, outputs['prior'])) / batch_size
        logging_output['kld'] = kld.item()

        tc = total_correlation(outputs['z'], posterior) / batch_size
        logging_output['tc'] = tc.item()

        x_recon = outputs['x']
        # p(x|z)
        rec = torch.sum(- x_recon.log_prob(sample['image'])) / batch_size
        logging_output['rec'] = rec.item()

        return (rec, kld, tc), batch_size, logging_output


def total_correlation(z, posterior):
    """estimate of total correlation on a batch"""

    # calculate log(q(z(x_j)|x_i)) for every sample in the batch
    # batch_size x batch_size x code_size
    log_probs = posterior.log_prob(
        z.unsqueeze(1).expand(-1, z.size(0), -1))

    log_qz  = torch.sum(
        torch.logsumexp(log_probs.sum(2), dim=1)
    )

    log_qzi = torch.sum(
        torch.logsumexp(log_probs, dim=1)
    )
    # note the constant is omitted
    return log_qz - log_qzi
