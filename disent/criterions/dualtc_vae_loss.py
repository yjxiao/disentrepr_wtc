import torch
from torch.nn.modules.loss import _Loss
from torch.distributions import kl_divergence


class DualTCVAELoss(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def loss_components(self):
        return ['rec', 'kld', 'dualtc', 'adv_loss']
    
    def forward(self, model, sample):
        outputs = model['main'](sample)
        batch_size = sample['batch_size']
        posterior = outputs['posterior']
        z = outputs['z']
        logging_output = {}
        logging_output['batch_size'] = batch_size
        
        # Rec
        x_recon = outputs['x']
        # p(x|z)
        rec = torch.sum(- x_recon.log_prob(sample['image'])) / batch_size
        logging_output['rec'] = rec.item()

        # KLD
        kld = torch.sum(kl_divergence(posterior, outputs['prior'])) / batch_size
        logging_output['kld'] = kld.item()

        # DualTC
        # inputs size: (batch_size, code_size)
        # first repeat input into (batch_size, code_size, code_size)
        code_size = z.size(1)
        # note: expand does not work
        inputs = z.unsqueeze(1).repeat(1, code_size, 1)
        # next step set all diagonal inputs to zero
        mask = torch.diag_embed(
            inputs.new_ones((batch_size, code_size), dtype=torch.uint8))
        inputs[mask] = 0
        cond_qzi = model['adversarial'](inputs)    # q(z_i|z_{-i})
        log_qzi = torch.sum(cond_qzi.log_prob(z))
        # (batch_size, batch_size, code_size)
        log_probs = posterior.log_prob(
            z.unsqueeze(1).expand(-1, z.size(0), -1))
        log_qz = torch.sum(
            torch.logsumexp(log_probs.sum(2), dim=1))
        # dualtc = E[sum(log(qzi))-log(qz)]
        dualtc = (log_qzi - log_qz) / batch_size
        logging_output['dualtc'] = dualtc.item()

        # adv_loss
        adv_loss = - log_qzi / batch_size
        logging_output['adv_loss'] = adv_loss.item()
        return (rec, kld, dualtc, adv_loss), batch_size, logging_output
