import numpy as np
import scipy.io as sio
import PIL

from . import VisionDataset



class Cars3D(VisionDataset):

    def __init__(self, root, split="train"):
        super().__init__(root)

        self.split = split

    def _load_data(self):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        all_files = [x for x in os.listdir(self.root) if '.mat' in x]
        for i, filename in enumerate(all_files):
            with open(os.path.join(self.root, filename), 'rb') as f:
                mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
            flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
            rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
            for i in range(flattened_mesh.shape[0]):
                pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
                pic.thumbnail((64, 64, 3), PIL.Image.ANTIALIAS)
                rescaled_mesh[i, :, :, :] = np.array(pic)
            data_mesh = rescaled_mesh * 1. / 255
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i, len(factor1) * len(factor2))
            ])
            
