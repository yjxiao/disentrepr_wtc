import os

import numpy as np
import scipy.io as sio
import PIL
import torch

from . import VisionDataset, register_dataset


@register_dataset('cars3d')
class Cars3D(VisionDataset):
    in_channels = 3
    
    def _load_data(self):
        images = np.zeros((183, 24 * 4, 3, 64, 64))
        factors = np.zeros((183, 24 * 4, 3), dtype=int)
        factor_dims = [24, 4]    # azimuth and elevation
        factor_ranges = [np.arange(v) for v in factor_dims]

        all_files = [x for x in os.listdir(self.root) if '.mat' in x]
        assert len(all_files) == 183
        for i, filename in enumerate(all_files):
            # read data as size (24, 4, 128, 128, 3)
            with open(os.path.join(self.root, filename), 'rb') as f:
                mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
            # flatten to (24 x 4, 128, 128, 3)
            flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
            rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
            for j in range(flattened_mesh.shape[0]):
                pic = PIL.Image.fromarray(flattened_mesh[j, :, :, :])
                pic.thumbnail((64, 64, 3), PIL.Image.ANTIALIAS)
                rescaled_mesh[j, :, :, :] = np.array(pic)
            # resize to (24 x 4, 64, 64, 3) and rescale
            data_mesh = rescaled_mesh * 1. / 255

            factor_i = np.stack(
                np.meshgrid([i], *factor_ranges, indexing='ij'), axis=-1)
            
            factors[i] = factor_i.reshape((-1, 3))
            # transpose to (24 x 4, 3, 64, 64)
            images[i] = np.einsum("abcd->adbc", data_mesh)
            
        images = images.reshape((-1, 3, 64, 64))
        factors = factors.reshape((-1, 3))
        self.images = torch.from_numpy(images).float()
        self.factors = torch.from_numpy(factors)
        self._image_dims = tuple(images.shape[1:])

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        item = {
            'id': index,
            'factor': self.factors[index],
            'image': self.images[index]
        }
        return item

    @property
    def image_dims(self):
        return self._image_dims

    @property
    def factor_dims(self):
        return (183, 24, 4)

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
