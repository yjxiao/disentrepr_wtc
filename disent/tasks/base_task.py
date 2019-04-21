from disent.data import build_dataset, data_utils
from disent.data import EpochBatchIterator


class BaseTask(object):
    """Base class for all tasks. Mainly handles data loading. """

    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, args):
        self.args = args

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self):
        self.dataset = build_dataset(args, True)
        print('| {} {} examples'.format(
            self.args.dataset, len(self.dataset)))

    def get_batch_iterator(self, dataset, batch_size, seed=1):
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        data_utils.batch_sampler(indices, batch_size=batch_size)
        return EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed)
