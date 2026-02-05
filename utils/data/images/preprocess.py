"""
Image Preprocessing Utilities

Functions for converting and sampling from images for point cloud generation.
"""
import torch


def sample_image(image, N=1000, D=2):
    """
    Sample points from an image with probability proportional to pixel intensity.

    Uses rejection sampling where acceptance probability is based on normalized pixel values. Denser sampling occurs
    in brighter regions.

    Args:
        image: Grayscale image tensor, shape (H, W)
        N: Number of points to sample
        D: Dimension (should be 2 for images)

    Returns:
        torch.Tensor: Sampled points in image coordinates, shape (N, 2)

    """
    shape = torch.tensor(image.shape)

    normalized_image = image / image.max()
    points = torch.tensor([])

    while points.shape[0] < N:
        random_points = torch.rand(size=(N, D), dtype=torch.float32) * shape
        pixel_points = torch.floor(random_points).type(torch.int64)

        accepted = torch.rand(N) < normalized_image[pixel_points[:, 0], pixel_points[:, 1]]  # TODO gérer dimension
        points = torch.concatenate([points, random_points[accepted]])

    return points[:N]


def convert_grayscale(image, D=2):
    """
    Convert an image to grayscale.

    Handles multiple input formats:
    - Already grayscale: returns as it is
    - RGB (3 channels): averages channels
    - RGBA (4 channels): combines RGB average with alpha channel

    Args:
        image: Input image tensor
        D: Spatial dimension (2 for standard images)

    Returns:
        torch.Tensor: Grayscale image, shape (H, W)
    """
    if len(image.shape) == D:
        return image
    elif image.shape[-1] == 3:
        return image.mean(dim=D)
    else:
        return (1 - image[..., :3].mean(dim=D)) * image[..., -1].squeeze()


def convert_mask(image):
    """
    Convert grayscale image to binary mask.

    Pixels above half the maximum intensity are considered as "True", others "False".

    Args:
        image: Grayscale image tensor

    Returns:
        torch.Tensor: Binary mask (boolean tensor), same shape as input
    """
    return image > (image.max() / 2)
