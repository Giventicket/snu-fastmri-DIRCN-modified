import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import fastmri
from ...preprocessing import KspaceToImage
from .resxunet import ResXUNet

import matplotlib.pyplot as plt

class NormNet(nn.Module):
    """A normalized wrapper for ResXUNet or for any models in that manner.
    Each input channel is normalized independently, and the means and stds of the first channel, i.e.,
    for the last cascade is used to de-normalize the output prediction for said cascade.
    """

    def __init__(self,
                 n: int = 24,
                 n_channels: int = 2,
                 groups: int = 4,
                 bias: bool = True,
                 ratio: float = 1. / 8,
                 interconnections: bool = False,
                 make_interconnections: bool = False,
                 ):
        """
        Args:
            n (int): the number of channels in the version3
            n_channels (int): the number of input channels
            groups (int): the number of groups used in the convolutions, needs to dividable with n
            bias (bool): whether to use bias or not
            ratio (float): the ratio for squeeze and excitation
            interconnections (bool): whether to enable interconnection
            make_interconnections (bool): whether to make the version3 accept interconnection input
        """
        super().__init__()

        self.interconnections = interconnections

        self.model = ResXUNet(
            n_channels=n_channels,
            n_classes=2,
            n=n,
            groups=groups,
            bias=bias,
            ratio=ratio,
            activation=torch.nn.SiLU(inplace=True),
            interconnections=interconnections,
            make_interconnections=make_interconnections
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        a = x.permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w)
        return a

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, c, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize each channel individually
        b, c, h, w = x.shape # (B, 2 * C, H, W)
        x = x.view(b, c, h * w) # (B, 2 * C, H * W)

        mean = x.mean(dim=2).view(b, c, 1, 1)  # (B, 2 * C, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1) # (B, 2 * C, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(
            self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape

        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
            self,
            x: torch.Tensor,
            h_pad: List[int],
            w_pad: List[int],
            h_mult: int,
            w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, internals: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        # get shapes for the version3 and normalize
        x = self.complex_to_chan_dim(x) # (B, 2 * C, H, W)
        x, mean, std = self.norm(x) # (B, 2 * C, H, W), (B, 2 * C, 1, 1), (B, 2 * C, 1, 1)
        x, pad_sizes = self.pad(x) # (B, 2 * C, 400, 400)

        if self.interconnections:
            x, internals = self.model(x, internals) # (B, 2 * C, 400, 400)
        else:
            x = self.model(x)  # (B, 2 * C, 400, 400)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)  # (B, 2 * C, H, W)
        x = self.unnorm(x, mean, std)  # (B, 2 * C, H, W)
        x = self.chan_complex_to_last_dim(x)  # (B, C, H, W, 2)
        if self.interconnections:
            return x, internals
        return x



class NormNet_image(nn.Module):
    """
    NormNet_image
    """

    def __init__(self,
                 n: int,
                 groups: int,
                 bias: bool = True,
                 ratio: float = 1. / 8,
                 ):
        """
        Args:
            n (int): the number of channels in the version3
            groups (int): the number of groups used in the convolutions, needs to dividable with n
            bias (bool): whether to use bias or not
            ratio (float): the ratio for squeeze and excitation
        """
        super().__init__()

        self.model = ResXUNet(
            n_channels=1,
            n_classes=1,
            n=n,
            groups=groups,
            bias=bias,
            ratio=ratio,
            activation=torch.nn.SiLU(inplace=True),
            interconnections=False,
            make_interconnections=False
        )

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize each channel individually
        b, c, h, w = x.shape # (B, C, H, W)
        x = x.view(b, c, h * w) # (B, C, H * W)

        mean = x.mean(dim=2).view(b, c, 1, 1)  # (B, C, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1) # (B, C, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(
            self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape

        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
            self,
            x: torch.Tensor,
            h_pad: List[int],
            w_pad: List[int],
            h_mult: int,
            w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, internals: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        # get shapes for the version3 and normalize
        x, mean, std = self.norm(x) # (B, C, H, W), (B, C, 1, 1), (B, C, 1, 1)
        x, pad_sizes = self.pad(x) # (B, C, 400, 400)

        x = self.model(x)  # (B, C, 400, 400)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)  # (B, C, H, W)
        x = self.unnorm(x, mean, std)  # (B, C, H, W)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This version3 applies an IFFT to multichannel k-space data and then a U-Net based version3
    to the coil images to estimate coil sensitivities.
    """

    def __init__(
            self,
            sense_n: int = 12,
            sense_groups: int = 1,
            bias: bool = True,
            ratio: float = 1. / 8,
    ):
        """
        Args:
            sense_n (int, optional): the number of channels to use in the sense network. Defaults to 12.
            sense_groups (int, optional): the number of groups to use in the sense network. Defaults to 3.
            bias (bool, optional): wheter to use bias or not. Defaults to True.
            ratio (float, optional): the ratio to use in squeeze and excitation. Defaults to 1./8.
        """
        super().__init__()

        self.model = NormNet(
            n=sense_n,
            groups=sense_groups,
            bias=bias,
            ratio=ratio,
            interconnections=False,
            make_interconnections=False,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.view(b * c, 1, h, w, two), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, two = x.shape
        assert two == 2
        c = bc // batch_size

        return x.view(batch_size, c, h, w, two)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def mask_center(self, x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
        """
        Initializes a mask with the center filled in.
        Args:
            mask_from: Part of center to start filling.
            mask_to: Part of center to end filling.
        Returns:
            A mask with the center filled.
        """
        mask = torch.zeros_like(x)
        mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
        return mask

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # ACS mask
        cent = mask.shape[-2] // 2
        left = torch.nonzero(mask.squeeze()[:cent] == 0)[-1]
        right = torch.nonzero(mask.squeeze()[cent:] == 0)[0] + cent
        num_low_freqs = right - left
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        x = self.mask_center(masked_kspace, pad, pad + num_low_freqs)

        # IFT
        x = fastmri.ifft2c(x) # (B, C, H, W, 2)
        x, b = self.chans_to_batch_dim(x)  # (C, B, H, W, 2)

        # estimate sensitivities
        x = self.model(x) # (C, B, H, W, 2)
        x = self.batch_chans_to_chan_dim(x, b) # (B, C, H, W)
        x = self.divide_root_sum_of_squares(x) # (B, C, H, W)

        return x


class DIRCN(nn.Module):
    """
    The full Densly interconnected residual cascading network (DIRCN)
    """

    def __init__(
            self,
            num_cascades: int = 30,
            n: int = 20,
            sense_n: int = 12,
            groups: int = 4,
            sense_groups: int = 1,
            bias: bool = True,
            ratio: float = 1. / 8,
            dense: bool = True,
            variational: bool = False,
            interconnections: bool = True,
    ):
        """Init for the DIRCN

        Args:
            num_cascades (int, optional): the total number of cascades. Defaults to 30.
            n (int, optional): the number of channels to use in each cascade. Defaults to 20.
            sense_n (int, optional): the number of channels in the sense network. Defaults to 12.
            groups (int, optional): the number of groups for each cascade network. Defaults to 4.
            sense_groups (int, optional): the number of groups in the sense network. Defaults to 1.
            bias (bool, optional): whether to use bias or not. Defaults to True.
            ratio (float, optional): the ratio for squeeze and excitation. Defaults to 1./8.
            dense (bool, optional): whether to use dense connections or not. Defaults to True.
            variational (bool, optional): changes the data consistency to a variational update mechanism. Defaults to False.
            interconnections (bool, optional): whether to use interconnections. Defaults to True.
        """
        super().__init__()
        self.interconnections = interconnections
        self.sens_net = SensitivityModel(
            sense_n=sense_n,
            sense_groups=sense_groups,
            bias=bias,
            ratio=ratio,
        )
        self.i_cascades = nn.ModuleList(
            [ImageBlock(
                dense=dense,
                interconnections=interconnections,
                variational=variational,
                i_model=NormNet(
                    n=n,
                    # n_channels=2 * (i + 1) if dense else 2,
                    groups=groups,
                    bias=bias,
                    ratio=ratio,
                    interconnections=interconnections,
                    make_interconnections=True if interconnections and i > 0 else False,
                )) for i in range(num_cascades)])

    def forward(
            self,
            masked_kspace: torch.Tensor,
            mask: torch.Tensor = None,
    ) -> torch.Tensor:
        kspace_pred = masked_kspace.clone() # (B, C, H, W, 2)
        sens_maps = self.sens_net(masked_kspace, mask) # (B, C, H, W)
        sens_maps = torch.view_as_complex(sens_maps) # (B, C, H, W, 2)
        sens_conj = torch.conj(sens_maps) # (B, C, H, W, 2)

        i_concat = None  # The first concat is always None
        interconnections = None
        for idx, i_cascade in enumerate(self.i_cascades):
            if self.interconnections:
                kspace_pred, i_concat, interconnections = i_cascade(kspace_pred, masked_kspace, mask, sens_maps,
                                                                    sens_conj, i_concat, interconnections)
            else:
                kspace_pred, i_concat = i_cascade(kspace_pred, masked_kspace, mask, sens_maps, sens_conj, i_concat)

        del sens_maps, sens_conj, mask, i_concat, interconnections
        image_pred = fastmri.ifft2c(kspace_pred) # (B, C, H, W, 2)
        image_pred = fastmri.complex_abs(image_pred)  # (B, C, H, W)
        image_pred = fastmri.rss(image_pred, dim=1)  # (B, H, W)

        return image_pred


class ImageBlock(nn.Module):
    """
    Model block for end-to-end variational network.
    This version3 applies a combination of soft data consistency with the input
    version3 as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self,
                 i_model: nn.Module,
                 dense: bool = True,
                 variational: bool = False,
                 interconnections: bool = False,
                 ):
        """
        Args:
            version3: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.i_model = i_model
        self.dense = dense
        self.variational = variational
        self.interconnections = interconnections
        self.dc_weight = nn.Parameter(torch.Tensor([0.01]))
        self.cat_weight = nn.Parameter(torch.Tensor([1.00])) ## for version 2

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_conj: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x) # (B, C, H, W, 2)
        return fastmri.complex_mul(x, sens_conj).sum(
            dim=1, keepdim=True
        ) # (B, 1, H, W, 2)

    def forward(
            self,
            current_kspace: torch.Tensor,
            ref_kspace: torch.Tensor,
            mask: torch.Tensor,
            sens_maps: torch.Tensor,
            sens_conj: torch.Tensor,
            concat: torch.Tensor,
            interconnections: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        inp = self.sens_reduce(current_kspace, sens_conj) # (B, 1, H, W, 2)
        if concat is None or not self.dense:
            concat = inp # (B, 1, H, W, 2)
        else:
            concat = inp + concat * self.cat_weight ## for version 2
            # concat = (inp + concat * self.cat_weight)/(1 + self.cat_weight) ## for version 3

        if self.interconnections:
            image, interconnections = self.i_model(concat, interconnections) # (B, 1, H, W, 2), List
        else:
            image = self.i_model(concat) # (B, 1, H, W, 2)

        image_expanded = self.sens_expand(image, sens_maps) # (B, C, H, W, 2)

        # reduced_image = fastmri.complex_abs(image).squeeze(dim = 1)

        if self.variational:
            zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
            soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
            kspace = current_kspace - soft_dc - image_expanded
        else:
            kspace = torch.where(mask,
                                 (ref_kspace + self.dc_weight * image_expanded) / (1 + self.dc_weight),
                                 image_expanded)

        if not self.dense:
            concat = None
        if self.interconnections:
            return kspace, concat, interconnections
        return kspace, concat
