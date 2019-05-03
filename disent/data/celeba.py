import os
import numpy as np
import PIL

import torch

from . import VisionDataset, register_dataset
from .utils import download_file_from_google_drive, check_integrity


@register_dataset('celeba')
class CelebA(VisionDataset):
    in_channels = 3

    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]
    
    def _load_data(self):
        processed_file = os.path.join(self.root, "celeba_ndarray_64x64x3.npz")
        if os.path.exists(processed_file):
            images = np.load(processed_file)['imgs']
            print('| loaded from {}'.format(processed_file))
        else:
            if not self._check_integrity():
                print('| dataset not found or corrupted. redownloading.')
                self.download()
            filenames = np.loadtxt(
                os.path.join(self.root, 'list_eval_partition.txt'),
                usecols=0, dtype=str)
            images = np.zeros((len(filenames), 64, 64, 3), dtype=np.uint8)
            for i, fname in enumerate(filenames):
                pic = PIL.Image.open(os.path.join(self.root, 'img_align_celeba', fname))
                # crop to square
                pic = pic.crop((0, 20, 178, 198))
                pic.thumbnail((64, 64, 3), PIL.Image.ANTIALIAS)
                images[i, :, :, :] = np.array(pic, dtype=np.uint8)
            # 64 x 64 x 3 -> 3 x 64 x 64
            images = np.einsum("abcd->adbc", images)
            np.savez(processed_file, imgs=images)
            
        self.images = torch.from_numpy(images * 1. / 255).float()
        self.factors = torch.from_numpy(
            np.loadtxt(
                os.path.join(self.root, "list_attr_celeba.txt"),
                skiprows=2, usecols=range(1, 41), dtype=int)
        )
        self.ids = torch.from_numpy(
            np.loadtxt(
                os.path.join(self.root, "identity_CelebA.txt"),
                usecols=1, dtype=int)
        )
        
    def _build_ft2idx(self):
        raise NotImplementedError("CelebA does not support factor to index mapping")
        
    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, filename)
            _, ext = os.path.splitext(filename)

            if ext != '.zip' and not check_integrity(fpath, md5):
                return False

        return os.path.isdir(os.path.join(self.root, 'img_align_celeba'))

    def download(self):
        import zipfile

        if self._check_integrity():
            print('| data files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, self.root, filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, "img_align_celeba.zip"), "r") as f:
            f.extractall(self.root)
            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = {
            'id': index,
            'factor': self.factors[index],
            'image': self.images[index]
        }
        return item

    @property
    def num_factors(self):
        raise NotImplementedError("CelebA does not support factor retrieval")

    @property
    def image_dims(self):
        return tuple(self.images.shape[1:])

    @property
    def factor_dims(self):
        raise NotImplementedError("CelebA does not support factor retrieval")

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
