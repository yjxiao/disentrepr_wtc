import os
import h5py
import numpy as np

import torch

from . import VisionDataset, register_dataset


@register_dataset('shapes3d')
class Shapes3D(VisionDataset):
    in_channels = 3
    
    def _load_data(self):
        filepath = os.path.join(self.root, '3dshapes.h5')
        data = h5py.File(filepath, 'r')
        images = np.einsum('abcd->adbc', data['images']) * 1. / 255
        self.images = torch.from_numpy(images).float()
        self._image_dims = tuple(self.images.shape[1:])
        # this will create a list of aranges to be used in meshgrid
        fvals = [torch.arange(v) for v in self.factor_dims]
        self.factors = torch.stack(torch.meshgrid(fvals), dim=-1).reshape(-1, self.num_factors)
        
    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        item = {
            'id': index,
            'factor': self.factors[index],
            'image': self.images[index],
        }
        return item

    @property
    def image_dims(self):
        return self._image_dims

    @property
    def factor_dims(self):
        return (10, 10, 10, 8, 4, 15)

    @property
    def num_factors(self):
        return len(self.factor_dims)
    
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
