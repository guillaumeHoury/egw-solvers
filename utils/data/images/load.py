"""
Image-Based Point Cloud Loading

Functions for loading and converting images into point clouds.
"""

import PIL.Image
import numpy as np
import torch

from utils.data.images.preprocess import sample_image, convert_mask, convert_grayscale
from utils.math.functions import noise, normalize, subsample


def load_image(path):
    """
    Load and normalize an image from file.

    Args:
       path: Path to image file (any format supported by PIL)

    Returns:
       torch.Tensor: Normalized image with values in [0, 1]
    """
    image = torch.tensor(np.array(PIL.Image.open(path)), dtype=torch.float32)
    return image / torch.max(image)


def load_pointcloud_from_image(path, D=2, mode='sampling', N=None):
    """
    Convert an image into a point cloud.

    Two modes are available:
    - 'sampling': Sample points with probability proportional to pixel intensity
    - 'pixels': Use pixel coordinates directly (where intensity > 0.5 * intensity.max())


    Args:
        path: Path to image file
        D: Dimension (2 for standard images)
        mode: 'sampling' for probabilistic sampling or 'pixels' for direct conversion
        N: Number of points. Required for 'sampling', optional for 'pixels'

    Returns:
        torch.Tensor: Point cloud representation of image, shape (N, D)
    """
    image = convert_grayscale(load_image(path))

    if mode == 'sampling':
        points = sample_image(image, N, D)
    elif mode == 'pixels':
        mask = convert_mask(image)
        points = torch.argwhere(mask).type(torch.float32)
        if N is not None: points = subsample(points, N)

    return normalize(points)


def load_grid_from_image(path, L=1, sigma=0.1):
    """
    Extract grid points from an image with optional noise.

    Samples points on a regular grid where the image has high intensity, then adds Gaussian noise for perturbation.

    Args:
        path: Path to image file
        L: Grid spacing (1 = every pixel, 2 = every other pixel, etc.)
        sigma: Standard deviation of Gaussian noise to add

    Returns:
        torch.Tensor: Noisy grid points from bright image regions, shape (K, 2)

    Note:
        Y-coordinates are flipped (negated) to match standard coordinate system where y increases upward.
    """
    image = convert_grayscale(load_image(path))

    grid = torch.meshgrid([torch.arange(0, image.shape[0], L), torch.arange(0, image.shape[1], L)], indexing='xy')
    points = torch.argwhere(image[grid[0], grid[1]]).to(dtype=torch.float32)
    points[:, 1] = -points[:, 1]
    points = noise(normalize(points), std=sigma).contiguous()
    return points
