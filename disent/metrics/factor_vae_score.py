from collections import OrderedDict

import numpy as np
import torch.nn as nn

from disent import utils
from disent.models import BaseModel
from . import BaseMetric, register_metric


@register_metric('factor')
class FactorVAEScore(BaseMetric):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--variance-estimate-batches', '--ve', type=int, metavar='N',
            help='number of samples to estimate global variance')
        parser.add_argument('--train-batches', type=int, metavar='N',
                            help='number of training batches')
        parser.add_argument('--eval-batches', type=int, metavar='N',
                            help='number of evaluation batches')

    def evaluate(self, task, model, seed):
        model = _check_model(model)
        model.eval()
        # size: (code_size,)
        global_variances = _estimate_variances(
            task, model, self.args.variance_estimate_batches,
            self.args.batch_size, seed, self.cuda)
        active_dims = _prune_dims(global_variances)
        stats = OrderedDict()

        if not active_dims.any():
            stats["train_acc"] = 0.
            stats["eval_acc"] = 0.
            stats["num_active_dims"] = 0
            return stats
        
        train_votes = _gather_votes(
            task, model, self.args.train_batches, self.args.batch_size,
            global_variances, active_dims, seed, self.cuda)
        classifier = np.argmax(train_votes, axis=0)
        other_index = np.arange(train_votes.shape[1])
        train_acc = np.sum(
            train_votes[classifier, other_index]) * 1. / np.sum(train_votes)
        
        eval_votes = _gather_votes(
            task, model, self.args.eval_batches, self.args.batch_size,
            global_variances, active_dims, seed + task.dataset.num_factors, self.cuda)
        eval_acc = np.sum(
            eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)

        stats['train_acc'] = train_acc
        stats['eval_acc'] = eval_acc
        stats['num_active_dims'] = sum(active_dims)
        return stats

        
def _gather_votes(task, model, num_batches, batch_size,
                  global_variances, active_dims, seed, cuda):
    num_factors = task.dataset.num_factors
    num_batches = int(num_batches // num_factors) + 1
    code_size = global_variances.shape[0]
    votes = np.zeros((num_factors, code_size), dtype=int)
    for factor_index in range(num_factors):
        itr = task.get_iterator_with_fixed_factor(
            dataset=task.dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            factor_index=factor_index,
            seed=seed+1+factor_index).next_epoch_itr(shuffle=False)
        
        for batch in itr:
            if cuda:
                batch = utils.move_to_cuda(batch)
            outputs = model(batch)
            reprs = outputs['z'].detach().cpu().numpy()
            local_variances = np.var(reprs, axis=0, ddof=1)
            code_index = np.argmin(
                local_variances[active_dims] / global_variances[active_dims]
            )
            votes[factor_index, code_index] += 1
    return votes

    
def _estimate_variances(task, model, num_batches, batch_size, seed, cuda):
    epoch_iter = task.get_batch_iterator(
        dataset=task.dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        seed=seed).next_epoch_itr(shuffle=False)

    code_samples = []
    for batch in epoch_iter:
        if cuda:
            batch = utils.move_to_cuda(batch)
        outputs = model(batch)
        # z has size (batch_size, code_size)
        code_samples.append(outputs['z'].detach().cpu().numpy())

    code_samples = np.concatenate(code_samples, axis=0)
    return np.var(code_samples, axis=0, ddof=1)
    

def _prune_dims(variances, threshold=0.05):
    scale_z = np.sqrt(variances)
    return scale_z >= threshold

    
def _check_model(model):
    if isinstance(model, nn.ModuleDict):
        model = model['main']
    assert issubclass(model.__class__, BaseModel), \
        "model class needs to be a subclass of BaseModel"
    return model

