""" 
Note this dataset is the version from https://arxiv.org/pdf/1802.04942.pdf

"""
import os
import numpy as np

import torch

from . import VisionDataset, register_dataset


@register_dataset('faces3d')
class Faces3D(VisionDataset):
    in_channels = 1
    
    def _load_data(self):
        filepath = os.path.join(self.root, 'basel_face_renders.pth')
        images = torch.load(filepath).float().div(255)
        self._factor_dims = tuple(images.size()[:-2])
        self._image_dims = tuple(images.size()[-2:])
        # this will create a list of aranges to be used in meshgrid
        fvals = [torch.arange(v) for v in self._factor_dims]
        self.factors = torch.stack(torch.meshgrid(fvals), dim=-1).reshape(-1, self.num_factors)
        self.images = images.reshape(len(self), 1, *self._image_dims)
        
    def __len__(self):
        return np.prod(self._factor_dims)

    def __getitem__(self, index):
        item = {
            'id': index,
            'factor': self.factors[index],
            'image': self.images[index]
        }
        return item

    @property
    def num_factors(self):
        return len(self._factor_dims)

    @property
    def image_dims(self):
        return self._image_dims

    @property
    def factor_dims(self):
        return self._factor_dims

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
