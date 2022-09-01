import fastmri
import numpy as np
import torch
from .. import fastmri

class KspaceToImage(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given k-space data to the corresponding image using Fourier transforms
    """

    def __init__(self,
                 norm: str = 'ortho',
                 shifting: bool = True,
                 ):
        """
        Args:
            norm (str): normalization method used in the ifft transform,
                see doc for torch.fft.ifft for possible args
        """
        self.norm = norm
        self.shifting = shifting

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with shape
                                   (coils, rows, columns) i.e ndim=3
        Returns:
            torch.Tensor: The ifft transformed k-space to image with shape
                          (channels, rows, columns)
        """
        if isinstance(tensor, torch.Tensor):
            data = fastmri.ifftshift(tensor, dim=(-2, -1))
            data = torch.fft.ifftn(data, dim=(-2, -1), norm=self.norm)
            data = fastmri.fftshift(data, dim=(-2, -1))
            return data
        elif isinstance(tensor, np.ndarray):
            data = np.fft.ifftshift(tensor, axes=(-2, -1))
            data = np.fft.ifftn(data, axes=(-2, -1), norm=self.norm)
            data = np.fft.fftshift(data, axes=(-2, -1))
            return data
