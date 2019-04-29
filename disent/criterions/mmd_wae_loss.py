import torch
from torch.nn.modules.loss import _Loss

from .mmd_utils import mmd_rbf, mmd_imq


class MMDWAELoss(_Loss):
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
        return ['rec', 'div']
    
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

        # MMD
        prior_z = prior.rsample()
        prior_var = prior.variance.mean()
        mmd = self.mmd(z, prior_z, prior_var)
        logging_output['div'] = mmd.item()

        return (rec, mmd), batch_size, logging_output


