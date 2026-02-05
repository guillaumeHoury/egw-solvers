"""
Synthetic Point Cloud Generation

Functions for generating synthetic point clouds from various geometric shapes and distributions.
"""
import torch

from utils.data.meshes.preprocess import sample_mesh


def normal(N, D=2, radius=1):
    """
    Generate points from a Gaussian (normal) distribution.

    Args:
        N: Number of points to generate
        D: Dimension of the space
        radius: Standard deviation of the distribution

    Returns:
        torch.Tensor: Points sampled from N(0, radius^2 I), shape (N, D)
    """
    points = radius * torch.normal(torch.zeros(N, D), 1)

    return points


def rectangle(N, D=2, lengths=None):
    """
    Generate uniformly distributed points in a rectangle/hyperrectangle.

    Args:
        N: Number of points
        D: Dimension
        lengths: Side lengths for each dimension. If None, uses unit hypercube

    Returns:
        torch.Tensor: Points uniformly distributed in [0, lengths[i]], shape (N, D)
    """
    lengths = lengths if lengths is not None else torch.ones(D)
    points = torch.rand(size=(N, D), dtype=torch.float32) * lengths
    return points


def triangle_full(N, D=2, lengths=None):
    """
    Generate uniformly distributed points in a D-simplex.

    Creates points in the simplex defined by {x : sum(x_i) < 1, x_i >= 0}. Uses rejection sampling for uniform
    distribution.

    Args:
        N: Number of points
        D: Dimension
        lengths: Optional scaling factors for each coordinate

    Returns:
        torch.Tensor: Points uniformly distributed in simplex, shape (N, D)
    """
    points = torch.tensor([])

    while points.shape[0] < N:
        random_points = (torch.rand(size=(N, D), dtype=torch.float32))
        accepted = random_points.sum(-1) < 1
        points = torch.concatenate([points, random_points[accepted]])
    points = points[:N]

    if lengths is not None:
        points = points * lengths

    return points


def tetrahedron(N, lengths=None):
    """
    Sample points uniformly from a tetrahedron surface.

    Creates a tetrahedral mesh and samples points from it.

    Args:
        N: Number of points to sample
        lengths: If None, uses regular tetrahedron. If provided (length 3), creates tetrahedron with specified edge
                 lengths

    Returns:
        torch.Tensor: Points sampled from tetrahedron, shape (N, 3)
    """
    if lengths is None:
        pos = torch.normal(mean=torch.zeros(size=(4, 3)))
        pos = pos / pos.norm(dim=-1, keepdim=True)
    else:
        pos = torch.tensor([[0, 0, 0], [lengths[0], 0, 0], [0, lengths[1], 0], [0, 0, lengths[2]]])

    face = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return sample_mesh(mesh={'pos': pos, 'face': face}, N=N)


def cube(N, lengths=None):
    """
    Generate uniformly distributed points on the surface of a cube.

    Samples points on all six faces of the cube with uniform distribution across the entire surface.

    Args:
        N: Number of points
        lengths: Side lengths [Lx, Ly, Lz]. If None, uses unit cube

    Returns:
        torch.Tensor: Points on cube surface, shape (N, 3)
    """
    pos_2d = torch.rand(size=(N, 2), dtype=torch.float32)

    face_id = torch.randint(0, 3, size=(N,), dtype=torch.int64)

    mapping = torch.tensor([[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]],
                           dtype=torch.float32)

    points = torch.einsum('ik,ikd->id', pos_2d, mapping[face_id])
    reverse = torch.randint(0, 2, size=(N,), dtype=torch.int64) > 0.5
    points[reverse] = 1 - points[reverse]

    if lengths is not None:
        points = points * lengths

    return points


def ball(N, D=2, radius=1):
    """
    Generate uniformly distributed points inside a ball/hypersphere.

    Uses rejection sampling to ensure uniform distribution in the volume.

    Args:
        N: Number of points
        D: Dimension of the space
        radius: Radius of the ball

    Returns:
        torch.Tensor: Points uniformly distributed in ball, shape (N, D)
    """
    points = torch.tensor([])

    while points.shape[0] < N:
        random_points = 2 * radius * (torch.rand(size=(N, D), dtype=torch.float32) - 0.5)
        accepted = torch.linalg.norm(random_points, dim=-1) < radius
        points = torch.concatenate([points, random_points[accepted]])

    return points[:N]


def sphere(N, D=2, radius=1):
    """
    Generate uniformly distributed points on a sphere/hypersphere surface.

    Args:
        N: Number of points
        D: Dimension of ambient space (sphere is (D-1)-dimensional)
        radius: Radius of the sphere

    Returns:
        torch.Tensor: Points on sphere surface, shape (N, D)
    """
    points = radius * torch.normal(torch.zeros(N, D), 1)
    norms = torch.sqrt((points ** 2).sum(-1))
    return points / norms[:, None]


def grid(N, D=2, lengths=None):
    """
    Generate points on a regular D-dimensional grid.

    Creates a grid with approximately N points, equally spaced in each dimension.

    Args:
        N: Approximate number of points (actual count is M^D where M = N^(1/D))
        D: Dimension
        lengths: Scaling factors for each dimension. If None, grid in [0,1]^D

    Returns:
        torch.Tensor: Grid points, shape (M^D, D) where M = floor(N^(1/D))
    """
    M = int(N ** (1 / D))
    X = torch.meshgrid([torch.arange(M) / M] * D, indexing='xy')
    points = torch.stack([X_i.flatten() for X_i in X]).transpose(1, 0).contiguous()

    if lengths is not None:
        points = points * lengths[None, :]
    return points


def grid_triangle(N, D=2, lengths=None):
    """
    Generate grid points inside a simplex.

    Creates a regular grid, then filters to keep only points where sum(x_i) < 1.

    Args:
        N: Approximate number of points before filtering
        D: Dimension
        lengths: Optional scaling factors

    Returns:
        torch.Tensor: Grid points inside simplex, shape (K, D) where K <= N
    """
    points = grid(N, D=D)

    accepted = points.sum(-1) < 1
    points = points[accepted]

    if lengths is not None:
        points = points * lengths[None, :]
    return points
