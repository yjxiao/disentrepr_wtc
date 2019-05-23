from collections import OrderedDict

import numpy as np
from scipy.stats import wasserstein_distance
import torch.nn as nn

from disent import utils
from disent.models import BaseModel
from . import BaseMetric, register_metric


@register_metric("wmod")
class WassersteinModularity(BaseMetric):
    
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
        # normalize z to have scale 1
        zs /= zs.max(1, keepdims=True) - zs.min(1, keepdims=True)
        np.random.seed(seed)
        shuffled_zs = [np.random.permutation(z) for z in zs]
        # (num_factors, eval_batches x batch_size)
        ys = np.transpose(np.concatenate(ys, axis=0))
        wmis = []
        for factor_idx, yi in enumerate(ys):
            wmi = []
            # calculate W1 for each value of the given factor
            for y in np.unique(yi):
                mask = yi == y
                # code_size
                wmi.append(
                    [wasserstein_distance(
                        z[mask], shuffled_z[mask]) \
                     for z, shuffled_z in zip(zs, shuffled_zs)]
                )
            # this becomes (num_factors, code_size)
            wmis.append(np.mean(wmi, axis=0))

        # (code_size, num_factors)
        wmis = np.transpose(wmis)
        assert wmis.shape[0] == zs.shape[0]
        assert wmis.shape[1] == ys.shape[0]
        
        stats['modularity'] = modularity(wmis)
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

    
def _check_model(model):
    if isinstance(model, nn.ModuleDict):
        model = model['main']
    assert issubclass(model.__class__, BaseModel), \
        "model class needs to be a subclass of BaseModel"
    return model

