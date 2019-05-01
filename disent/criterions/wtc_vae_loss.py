import torch
from torch.nn.modules.loss import _Loss
from torch.distributions import kl_divergence

from .mmd_utils import mmd_rbf, mmd_imq


class WTCVAELoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.mmd_kernel == 'rbf':
            self.mmd = mmd_rbf
        elif args.mmd_kernel == 'imq':
            self.mmd = mmd_imq
        else:
            raise ValueError("Unsupported kernel type: " + args.mmd_kernel)
        
    @property
    def loss_components(self):
        return ['rec', 'kld', 'wtc']
    
    def forward(self, model, sample):
        outputs = model(sample)
        batch_size = sample['batch_size']
        prior = outputs['prior']
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
            kl_divergence(outputs['posterior'], prior)) / batch_size
        logging_output['kld'] = kld.item()
        
        # WTC
        prior_var = prior.variance.mean()
        shuffled_z = shuffle_code(z)
        wtc = self.mmd(z, shuffled_z, prior_var)
        logging_output['wtc'] = wtc.item()

        return (rec, kld, wtc), batch_size, logging_output


def shuffle_code(code):
    """Shuffle latent variables across the batch
    
    Args:
        code: [batch_size, code_size]
    """
    code = code.clone()
    shuffled = []
    bsz, csz = code.size()
    for i in range(csz):
        idx = torch.randperm(bsz)
        shuffled.append(code[idx][:, i])
    return torch.stack(shuffled, dim=1)
