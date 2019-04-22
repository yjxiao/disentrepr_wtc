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

    def _load_data(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices

    def collater(self, samples):
        raise NotImplementedError
