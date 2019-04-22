import os
import numpy as np

import torch

from . import VisionDataset, register_dataset


@register_dataset('dsprites')
class DSprites(VisionDataset):
    in_channels = 1
    
    def _load_data(self):
        filepath = os.path.join(self.root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(filepath, encoding='latin1')
        self.images = torch.from_numpy(data['imgs']).float()
        self.factors = torch.from_numpy(data['latents_classes'])
        self._image_dims = tuple(self.images.size()[1:])

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        item = {
            'id': index,
            'factor': self.factors[index],
            'image': self.images[index].unsqueeze(0)
        }
        return item

    @property
    def image_dims(self):
        return self._image_dims

    @property
    def factor_dims(self):
        return (1, 3, 6, 40, 32, 32)

    @property
    def num_factors(self):
        return 6
    
    def collater(self, samples):
        return collate(samples)


def collate(samples):
    if len(samples) == 0:
        return {}
    idx = torch.LongTensor([s['id'] for s in samples])
    factor = torch.stack([s['factor'] for s in samples], dim=0)
    image = torch.stack([s['image'] for s in samples], dim=0)
    
    batch = {
        'id': idx, 'batch_size': len(samples),
        'factor': factor, 'image': image
    }
    
    return batch
