from collections import OrderedDict

import numpy as np
import scipy
import torch.nn as nn

from disent import utils
from disent.models import BaseModel
from . import BaseMetric, register_metric


@register_metric('unsup')
class UnsupervisedScores(BaseMetric):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--eval-batches', type=int, metavar='N',
                            help='number of evaluation batches')
        
    def evaluate(self, task, model, seed):
        model = _check_model(model)
        model.eval()
        stats = OrderedDict()

        itr = task.get_batch_iterator(
            dataset=task.dataset,
            batch_size=self.args.batch_size,
            num_batches=self.args.eval_batches,
            seed=seed).next_epoch_itr(shuffle=False)

        zs = []
        for batch in itr:
            if self.cuda:
                batch = utils.move_to_cuda(batch)
            outputs = model(batch)
            # (batch_size, code_size)
            zs.append(outputs['z'].detach().cpu().numpy())

        # (code_size, eval_batches x batch_size)
        zs = np.transpose(np.concatenate(zs, axis=0))
        cov = np.cov(zs)

        stats['total_correlation'] = gaussian_total_correlation(cov)
        stats['wasserstein_correlation'] = gaussian_wasserstein_correlation(cov)
        stats['wasserstein_correlation_norm'] = stats['wasserstein_correlation'] \
                                                / np.sum(np.diag(cov))
        return stats


def gaussian_total_correlation(cov):
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])


def gaussian_wasserstein_correlation(cov):
    sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
    return 2 * np.trace(cov) - 2 * np.trace(sqrtm)


def _check_model(model):
    if isinstance(model, nn.ModuleDict):
        model = model['main']
    assert issubclass(model.__class__, BaseModel), \
        "model class needs to be a subclass of BaseModel"
    return model

