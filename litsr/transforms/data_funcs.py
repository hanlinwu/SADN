import random
from typing import Iterable, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from . import matlab_funcs

#####################
# convert functions #
#####################


def uint2single(img: np.ndarray) -> np.ndarray:
    """Convert uint8 to float32"""

    assert img.dtype == np.uint8
    return np.float32(img / 255.0)


def single2uint(img: np.ndarray) -> np.ndarray:
    """Convert float32 to uint8"""

    return np.uint8((img.clip(0, 1) * 255.0).round())


def single2tensor(img: np.ndarray) -> np.ndarray:
    """Convert float32 to tensor"""

    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def pil2tensor(
    imgs: Union[Image.Image, Iterable[Image.Image]], rgb_range: Optional[int] = "255"
) -> torch.tensor:
    """Convert a pillow image to tensor"""

    def _pil2tensor(img, rgb_range):
        img = np.array(img)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.0)
        return tensor

    if not isinstance(imgs, list):
        return _pil2tensor(imgs, rgb_range)
    else:
        return [_pil2tensor(img, rgb_range) for img in imgs]


def tensor2np(tensors, data_range=(0, 1), is_clip=True):
    """Convert a Tensor to np.array without changing the data range.

    Args:
        tensors ([list | tensor]): input tensor or tensor list
    """

    def _tensor2np(x):
        x = x.numpy().astype(np.float32)
        if is_clip:
            x = x.clip(data_range[0], data_range[1])
        return x.transpose(1, 2, 0)

    if isinstance(tensors, list):
        return [_tensor2np(x) for x in tensors]
    else:
        return _tensor2np(tensors)


def tensor2uint8(
    tensors: Union[torch.tensor, Iterable[torch.tensor]], rgb_range: Optional[int] = 255
):
    """Convert tensor (0, rgb_range) to numpy array"""
    to_list_flag = 0
    if not isinstance(tensors, list):
        to_list_flag = 1
        tensors = [tensors]

    def quantize(img, rgb_range):
        pixel_range = 255.0 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round()

    array = [
        np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        for tensor in tensors
    ]
    if to_list_flag:
        array = array[0]

    return array


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[: h - h_remainder, : w - w_remainder, ...]
    else:
        raise ValueError(f"Wrong img ndim: {img.ndim}.")
    return img


def random_crop(imgs, patch_size):
    if not isinstance(imgs, list):
        imgs = [imgs]

    input_type = "Tensor" if torch.is_tensor(imgs[0]) else "Numpy"

    if input_type == "Tensor":
        h, w = imgs[0].size()[-2:]
    else:
        h, w = imgs[0].shape[0:2]

    if h < patch_size or w < patch_size:
        raise ValueError(
            f"LQ ({h}, {w}) is smaller than patch size "
            f"({patch_size}, {patch_size}). "
        )

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop lq patch
    if input_type == "Tensor":
        imgs = [v[:, :, top : top + patch_size, left : left + patch_size] for v in imgs]
    else:
        imgs = [v[top : top + patch_size, left : left + patch_size, ...] for v in imgs]

    if len(imgs) == 1:
        return imgs[0]
    else:
        return imgs


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = "Tensor" if torch.is_tensor(img_gts[0]) else "Numpy"

    if input_type == "Tensor":
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ",
            f"multiplication of LQ ({h_lq}, {w_lq}).",
        )
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
            f"({lq_patch_size}, {lq_patch_size}). "
            f"Please remove {gt_path}."
        )

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == "Tensor":
        img_lqs = [
            v[:, :, top : top + lq_patch_size, left : left + lq_patch_size]
            for v in img_lqs
        ]
    else:
        img_lqs = [
            v[top : top + lq_patch_size, left : left + lq_patch_size, ...]
            for v in img_lqs
        ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == "Tensor":
        img_gts = [
            v[:, :, top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size]
            for v in img_gts
        ]
    else:
        img_gts = [
            v[top_gt : top_gt + gt_patch_size, left_gt : left_gt + gt_patch_size, ...]
            for v in img_gts
        ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def vflip(img):
    """vertical flip"""
    return cv2.flip(img, 0, img)


def random_vflip(img, p=0.5):
    """Random vertival flips"""
    if random.random() < p:
        return vflip(img)
    else:
        return img


def hflip(img):
    """horizontal flip"""
    return cv2.flip(img, 1, img)


def random_hflip(img, p=0.5):
    """Random horizontal flips"""
    if random.random() < p:
        return hflip(img)
    else:
        return img


def rot90(img):
    """rotate 90"""
    return img.transpose(1, 0, 2)


def augment(imgs, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        img = img.copy()
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if return_status:
        return imgs, (hflip, vflip, rot90)
    else:
        return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def resize_cv2(img, scale=None, size=None, mode="bicubic"):
    """Resize image use opencv

    Args:
        img (np.array): Input image.
        scale (float | None): scale factor.
        mode ('bicubic' | 'bilinear' | 'nearest'): interpolate method.

    Return:
        img
    """
    assert scale or size
    h, w = img.shape[0:2]
    if not size:
        size = (int(h * scale), int(w * scale))

    size = (size[1], size[0])

    interpolation = {
        "bicubic": cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }[mode]

    return cv2.resize(img, size, interpolation=interpolation)


def resize_matlab(img, scale=None, size=None, mode="bicubic"):
    """Resize function the same as matlab

    Args:
        img (Tensor | np.array): Input image.
        scale (float): scale factor.
        mode ('bicubic'): matlab function only support bicubic interpolation.

    """
    assert mode == "bicubic" and (scale or size)
    to_uint_flag = 0
    if img.dtype == np.uint8:
        to_uint_flag = 1
        img = uint2single(img)
    img = matlab_funcs.imresize(img, scale, size)
    if to_uint_flag:
        img = single2uint(img)
    return img


def resize_pillow(img, scale=None, size=None, mode="bicubic"):

    assert scale or size
    if type(img) == np.ndarray:
        h, w = img.shape[0:2]
    else:
        h, w = img.size[0:2]
    if not size:
        size = (int(h * scale), int(w * scale))

    size = (size[1], size[0])
    to_numpy_flag = 0
    if type(img) == np.ndarray:
        to_numpy_flag = 1
        img = Image.fromarray(img)

    interpolation = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }[mode]

    img = img.resize(size, resample=interpolation)
    if to_numpy_flag:
        img = np.array(img)

    return img


def normalize(
    tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False
) -> Tensor:
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "Input tensor should be a torch tensor. Got {}.".format(type(tensor))
        )

    if tensor.ndim < 3:
        raise ValueError(
            "Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = "
            "{}.".format(tensor.size())
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                dtype
            )
        )
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def denormalize(
    tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False
) -> Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "Input tensor should be a torch tensor. Got {}.".format(type(tensor))
        )

    if tensor.ndim < 3:
        raise ValueError(
            "Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = "
            "{}.".format(tensor.size())
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                dtype
            )
        )
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(std)
    return tensor
