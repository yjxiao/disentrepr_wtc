from PIL import Image
import numpy as np

import torch
import torch.nn as nn


class ImageGenerator(object):
    def __init__(self):
        pass

    @torch.no_grad()
    def generate(self, model, sample, modifications):
        if isinstance(model, nn.ModuleDict):
            model = model['main']
        model.eval()

        _, z = model.encode(sample)
        results = []
        if len(modifications) == 0:
            # 1 x B
            x = model.decode(z)
            results.append(x.mean)
        else:
            for mod in modifications:
                idx = mod['dim']
                values = mod['values']
                mod_results = []
                for val in values:
                    z_copy = z.clone()
                    z_copy[:, idx] = val
                    x = model.decode(z_copy)
                    mod_results.append(x.mean)
                # D x V x B  (D: number of modified dimensions; V: number of values per mod)
                results.append(mod_results)
            
        return np.array(convert_to_images(results), dtype=object)


def convert_to_images(inputs):
    """Input is a nested list of matrices of size BxCxHxW. """
    if isinstance(inputs, torch.Tensor):
        images = inputs.detach().cpu().numpy()
        results = []
        # this means a batch of images
        assert len(images.shape) == 4, \
            "Expect input shape (B, C, H, W)"
        num_channels = images.shape[1]
        assert num_channels in [1, 3], \
            "Image expect number of channels 1 or 3, got " + str(num_channels)

        # (B, C, H, W) -> (B, H, W, C)
        images = np.einsum("abcd->acdb", (images * 255).astype(np.uint8))
        if num_channels == 1:
            images = images.squeeze(-1)
        for image in images:
            results.append(Image.fromarray(image))
        return results
    
    else:
        results = []
        for inp in inputs:
            results.append(convert_to_images(inp))
        return results
