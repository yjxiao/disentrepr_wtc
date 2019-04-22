import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import kl_divergence


class FactorVAELoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, model, sample):
        outputs = model['main'](sample)
        batch_size = sample['batch_size']
        posterior = outputs['posterior']
        z = outputs['z']
        logging_output = {}
        logging_output['batch_size'] = batch_size
        
        # kld
        kld = torch.sum(kl_divergence(posterior, outputs['prior'])) / batch_size
        logging_output['kld'] = kld.item()

        # tc
        logits_qz = model['adversarial'](z)
        # TC = E[log(p_real)-log(p_fake)]
        tc = torch.sum(logits_qz[:, 0] - logits_qz[:, 1]) / batch_size
        logging_output['tc'] = tc.item()

        # rec
        x_recon = outputs['x']
        # p(x|z)
        rec = torch.sum(- x_recon.log_prob(sample['image'])) / batch_size
        logging_output['rec'] = rec.item()

        # adv_loss
        z_shuffled = shuffle_code(z.detach())
        logits_qz_shuffled = model['adversarial'](z_shuffled)
        adv_loss = 0.5 * (
            F.nll_loss(logits_qz, z.new_zeros(z.size(0)).long(), reduction='sum') +
            F.nll_loss(logits_qz_shuffled, z.new_ones(z.size(0)).long(), reduction='sum')
        ) / batch_size
        logging_output['adv_loss'] = adv_loss.item()
        return (rec, kld, tc, adv_loss), batch_size, logging_output


def shuffle_code(code):
    """Shuffle latent variables across the batch
    
    Args:
        code: [batch_size, code_size]
    """
    shuffled = []
    bsz, csz = code.size()
    for i in range(csz):
        idx = torch.randperm(bsz)
        shuffled.append(code[idx][:, i])
    return torch.stack(shuffled, dim=1)
