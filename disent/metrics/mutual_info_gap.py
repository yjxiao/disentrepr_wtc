from collections import OrderedDict

import numpy as np
from sklearn.metrics import mutual_info_score
import torch.nn as nn

from disent import utils
from disent.models import BaseModel
from . import BaseMetric, register_metric


@register_metric('mig')
class MutualInformationGap(BaseMetric):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--eval-batches', type=int, metavar='N',
                            help='number of evaluation batches')
        parser.add_argument('--num-bins', type=int, default=20,
                            help='number of bins to use when discretizing')
        
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
        ys = []
        for batch in itr:
            if self.cuda:
                batch = utils.move_to_cuda(batch)
            outputs = model(batch)
            # (batch_size, code_size)
            zs.append(outputs['z'].detach().cpu().numpy())
            # (batch_size, num_factors)
            ys.append(batch['factor'].detach().cpu().numpy())

        # (code_size, eval_batches x batch_size)
        zs = np.transpose(np.concatenate(zs, axis=0))
        # (num_factors, eval_batches x batch_size)
        ys = np.transpose(np.concatenate(ys, axis=0))
        
        discrete_zs = _discretize(zs, self.args.num_bins)
        # (code_size, num_factors)
        mi = _discrete_mutual_info(discrete_zs, ys)
        # (num_factors,)
        entropy = _discrete_entropy(ys)
        sorted_mi = np.sort(mi, axis=0)[::-1]
        mig = np.divide(sorted_mi[0, :] - sorted_mi[1, :], entropy[:])
        for i in range(len(mig)):
            stats['mig_factor_{}'.format(i)] = mig[i]
        stats['avg_mig'] = np.mean(mig[np.isfinite(mig)])
        return stats

    
def _discretize(target, num_bins):
    result = np.zeros_like(target)
    for i in range(target.shape[0]):    # code dimensions
        result[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return result


def _discrete_mutual_info(zs, ys):
    code_size = zs.shape[0]
    num_factors = ys.shape[0]
    mi = np.zeros([code_size, num_factors])
    for i in range(code_size):
        for j in range(num_factors):
            mi[i, j] = mutual_info_score(ys[j, :], zs[i, :])
    return mi

    
def _discrete_entropy(ys):
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[j, :], ys[j, :])
    return h

    
def _check_model(model):
    if isinstance(model, nn.ModuleDict):
        model = model['main']
    assert issubclass(model.__class__, BaseModel), \
        "model class needs to be a subclass of BaseModel"
    return model

