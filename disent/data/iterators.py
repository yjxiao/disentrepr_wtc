# Adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/data/iterators.py
import itertools

import numpy as np
import torch

from . import data_utils


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.
    Args:
        iterable (iterable): iterable to wrap
    Attributes:
        count (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable):
        self.iterable = iterable
        self.count = 0
        self.itr = iter(self)
        self.len = len(iterable)

    def __len__(self):
        return self.len

    def __iter__(self):
        for x in self.iterable:
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.count < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        self.len -= num_to_skip
        return self


class EpochBatchIterator(object):
    def __init__(self, dataset, collate_fn, batch_sampler, seed=42):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.frozen_batches = tuple(batch_sampler)
        self.seed = seed

        self.epoch = 0
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

    def __len__(self):
        return len(self.frozen_batches)

    def next_epoch_itr(self, shuffle=True):
        """Return a new iterator over the dataset.
        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle,
            )
        return self._cur_epoch_itr

    def end_of_epoch(self):
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            itr = self._get_iterator_for_epoch(self.epoch, state_dict.get('shuffle', True))
            if itr_pos < len(itr):
                self._next_epoch_itr = itr.skip(itr_pos)

    def _get_iterator_for_epoch(self, epoch, shuffle):
        if shuffle:
            batches = _shuffle_batches(
                list(self.frozen_batches), self.seed + epoch)
        else:
            batches = self.frozen_batches
            
        return CountingIterator(torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches))


def _shuffle_batches(batches, seed):
    # set seed based on the seed and epoch number so that we get
    # reproducible results when resuming from checkpoints
    with data_utils.numpy_seed(seed):
        np.random.shuffle(batches)
        return batches
