import os

import numpy as np
import torch.utils.data


class VisionDataset(torch.utils.data.Dataset):

    @classmethod
    def build_dataset(cls, args, shuffle=False):
        root = os.path.join(args.data_dir, args.dataset)
        return cls(root, shuffle)

    def __init__(self, root, shuffle=False):
        self.root = root
        self.shuffle = shuffle
        self._load_data()
        self._ft2idx = None    # factor to idx mapping
        
    def _load_data(self):
        raise NotImplementedError

    def _build_ft2idx(self):
        # there is a potential problem that some of the factors
        # have no examples associated
        if self._ft2idx is None:
            self._ft2idx = np.zeros(self.factor_dims, dtype=int)
            for idx, factor in enumerate(self.factors):
                self._ft2idx[tuple(factor.numpy())] = idx
    
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def factor_dims(self):
        raise NotImplementedError
    
    @property
    def num_factors(self):
        raise NotImplementedError
    
    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    def indices_from_factors(self, factors):
        # return indices corresponding to the given factors
        if len(factors) == 0 or factors is None:
            return []
        if self._ft2idx is None:
            self._build_ft2idx()
        # input size: (num_examples, num_factors)
        assert len(factors[0]) == self.num_factors, \
            'number of factors does not match: {} expected'.format(
                self.num_factors)
        return self._ft2idx[tuple(x for x in np.transpose(factors))]

    def sample_factors(self, sample_size):
        indices = np.random.choice(
            np.arange(len(self)), sample_size, replace=False)
        return self.factors.numpy()[indices]
    
    def collater(self, samples):
        raise NotImplementedError
