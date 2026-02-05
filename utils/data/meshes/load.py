"""
Mesh Loading and Conversion

Functions for loading 3D meshes from various file formats and converting them into torch point clouds.
"""
import pyvista as pv
import torch

from utils.data.meshes.preprocess import sample_mesh
from utils.math.functions import linear, subsample


def load_from_pv(path, subdivide=False, nsub=1):
    """
    Load a 3D mesh using PyVista and optionally subdivide for refinement.

    The mesh is cleaned (removes duplicate points, degenerate cells) and can be subdivided using Loop subdivision.

    Args:
        path: Path to mesh file (any format supported by PyVista)
        subdivide: If True, apply subdivision to increase mesh resolution
        nsub: Number of subdivision iterations (more = smoother/denser mesh)

    Returns:
        dict: Mesh dictionary containing:
            - 'pos': Vertex positions, shape (V, 3)
            - 'face': Triangle faces as vertex indices, shape (F, 3)
    """
    pv_mesh = pv.read(path).clean()

    if subdivide:
        pv_mesh = pv_mesh.subdivide(nsub=nsub, subfilter='loop')

    pos = torch.tensor(pv_mesh.points, dtype=torch.float32)

    face = torch.tensor(pv_mesh.faces.reshape(-1, 4)[:, 1:], dtype=torch.int64)

    return {'pos': pos, 'face': face}


def load_off(path, **kwargs):
    """
    Load a mesh from OFF (Object File Format) file.

    Args:
        path: Path to .off file

    Returns:
        dict: Mesh dictionary containing:
            - 'pos': Vertex positions, shape (V, 3)
            - 'face': Triangle faces as vertex indices, shape (F, 3)
    """
    file = open(path)
    if 'OFF' != file.readline().strip():
        raise ('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]

    return {'pos': torch.tensor(verts, dtype=torch.float32), 'face': torch.tensor(faces, dtype=torch.int64)}


def load_mesh(path, extension='auto', **kwargs):
    """
    Load a 3D mesh with automatic format detection.

    Dispatches to appropriate loader based on file extension.
    Uses custom OFF loader for .off files, PyVista for everything else.

    Args:
       path: Path to mesh file
       extension: File extension ('off', 'ply', 'obj', etc.) or 'auto' to detect
                 from filename (uses last 3 characters)
       **kwargs: Additional arguments passed to specific loader
                (e.g., subdivide, nsub for load_from_pv)

    Returns:
        dict: Mesh dictionary containing:
            - 'pos': Vertex positions, shape (V, 3)
            - 'face': Triangle faces as vertex indices, shape (F, 3)
    """
    if extension == 'auto':
        extension = path[-3:]

    if extension == 'off':
        return load_off(path, **kwargs)
    else:
        return load_from_pv(path, **kwargs)


def load_pointcloud_from_mesh(path, mode='sampling', extension='auto', N=1024):
    """
    Convert a 3D mesh into a point cloud.

    Two conversion modes:
        - 'sampling': Sample N points uniformly from mesh surface (area-weighted)
        - 'vertices': Use mesh vertices directly as points

    After conversion, applies a coordinate transformation to reorient the mesh (switch Y and Z axes, and reverse their
    orientations).

    Args:
        path: Path to mesh file
        mode: Conversion mode - 'sampling' or 'vertices'
        extension: File format ('auto', 'off', 'ply', etc.)
        N: Number of points. Required for 'sampling', optional for 'vertices'

    Returns:
        torch.Tensor: Point cloud, shape (N, 3)

    """
    mesh = load_mesh(path, extension=extension)
    if mode == 'sampling':
        points = sample_mesh(mesh, N=N)
    elif mode == 'vertices':
        points = mesh['pos']
        if N is not None: points = subsample(points, N)

    return linear(points, A=torch.tensor([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float32))
