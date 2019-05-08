from collections import OrderedDict

import numpy as np
from sklearn.metrics import mutual_info_score
import torch.nn as nn

from disent import utils
from disent.models import BaseModel
from . import BaseMetric, register_metric


@register_metric("modularity")
class Modularity(BaseMetric):
    
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
        assert mi.shape[0] == zs.shape[0]
        assert mi.shape[1] == ys.shape[0]
        
        stats['modularity'] = modularity(mi)
        return stats

    
def modularity(mutual_info):
    squared_mi = np.square(mutual_info)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] -1.)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)

    
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

    
def _check_model(model):
    if isinstance(model, nn.ModuleDict):
        model = model['main']
    assert issubclass(model.__class__, BaseModel), \
        "model class needs to be a subclass of BaseModel"
    return model

