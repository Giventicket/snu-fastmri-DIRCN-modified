import numpy as np
import torch

from .kspace_to_image import KspaceToImage
from .image_to_kspace import ImageToKspace
from .crop_image import CropImage

class DownsampleFOV(object):
    """
    ***torchvision.Transforms compatible***

    Downsamples the FOV by fourier than cropping by inverse fouirer
    """
    def __init__(self, k_size: int = 320, i_size: int = 320):
        """
        Args:
            dim (int): the dimension for downsampling, 1 for height and 2 for width
            size (int): the length of k-space along the dim direction
        """
        self.k_size = k_size
        self.i_size = i_size

    def _numpy(self, tensor: np.ndarray):
        fft = KspaceToImage(norm='ortho')
        ifft = ImageToKspace(norm='ortho')
        i_crop = CropImage((self.i_size, self.i_size))
        k_crop = CropImage((self.k_size, self.k_size))

        tensor = fft(tensor)
        tensor = i_crop(tensor)
        tensor = ifft(tensor)
        if not self.i_size == self.k_size:
            tensor = k_crop(tensor)
        return tensor

    def __call__(self, tensor: np.ndarray):
        """
        Calculates the phase images
        Args:
            tensor (torch.Tensor): Complex Tensor of the image data with shape
                                   (coils, rows, columns, ...) i.e ndim >= 1
        Returns:
            torch.Tensor: The real phase images with equal shape

        """
        self.i_size = min(tensor.shape[-1], tensor.shape[-2])
        self.k_size = self.i_size
        return self._numpy(tensor)
