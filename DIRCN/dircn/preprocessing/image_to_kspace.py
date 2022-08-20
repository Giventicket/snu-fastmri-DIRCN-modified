import numpy as np

class ImageToKspace(object):
    """
    ***torchvision.Transforms compatible***

    Transforms the given image data to the corresponding image using Fourier transforms
    """
    def __init__(self,
                 norm: str = 'ortho',
                 ):
        """
        Args:
            norm (str): normalization method used in the fft transform,
                see doc for torch.fft.fft for possible args
        """
        self.norm = norm

    def __call__(self, tensor: np.ndarray):
        """
        Args:
            tensor (np.ndarray): Tensor of the image data with shape
                                   (coils, rows, columns) i.e ndim=3
        Returns:
            torch.Tensor: The fft transformed image to k-space with shape
                          (channels, rows, columns)

        """
        numpy_dtype = tensor.dtype
        tensor = tensor.astype(np.complex128)
        data = np.fft.fftshift(tensor, axes=(-2, -1))
        data = np.fft.fftn(data, axes=(-2, -1), norm=self.norm)
        data = np.fft.ifftshift(data, axes=(-2, -1))
        return data.astype(numpy_dtype)
