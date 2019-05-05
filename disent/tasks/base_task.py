import numpy as np

import torch

from disent.data import build_dataset, data_utils
from disent.data import EpochBatchIterator
from disent.generator import ImageGenerator


class BaseTask(object):
    """Base class for all tasks. Mainly handles data loading. """

    @staticmethod
    def add_args(parser):
        parser.add_argument('--data-dir', default='data/', type=str,
                            help='data directory')
        parser.add_argument('--dataset', default='dsprites', type=str,
                            help='dataset name to load')

    def __init__(self, args):
        self.args = args

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self):
        self.dataset = build_dataset(self.args, True)
        print('| {} {} examples'.format(
            self.args.dataset, len(self.dataset)))

    def build_generator(self, args):
        return ImageGenerator()
    
    def get_batch_iterator(self, dataset, batch_size,
                           num_batches=None,
                           seed=1):
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        batch_sampler = data_utils.batch_sampler(
            indices, batch_size=batch_size, num_batches=num_batches)
        
        return EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed)

    def get_iterator_with_fixed_factor(
            self, dataset, batch_size, num_batches,
            factor_index, seed):
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        result_indices = []
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            idx = indices[start:end]
            # (batch_size, num_factors)
            factors = dataset.factors.numpy()[idx]
            factors[:, factor_index] = factors[0, factor_index]
            result_indices.append(dataset.indices_from_factors(factors))

        indices = np.concatenate(result_indices)
        batch_sampler = data_utils.batch_sampler(
            indices, batch_size=batch_size, num_batches=num_batches)
        
        return EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed)
        
    def gen_step(self, generator, model, sample, modifications):
        with torch.no_grad():
            return generator.generate(model, sample, modifications)
