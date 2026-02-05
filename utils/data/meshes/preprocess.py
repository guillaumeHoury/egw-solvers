"""
Mesh Preprocessing Utilities

Functions for sampling points from mesh surfaces and simplifying meshes.
"""
import fast_simplification
import torch

from pykeops.torch import LazyTensor


def sample_mesh(mesh, N=1024):
    """
    Sample points uniformly from a triangular mesh surface.

    Args:
        mesh: Dictionary containing:
            - 'pos': Vertex positions, shape (M, D) where D is typically 3
            - 'face': Triangle face indices, shape (Q, 3)
        N: Number of points to sample

    Returns:
        torch.Tensor: Sampled points on mesh surface, shape (N, D)

    """
    pos = mesh['pos']  # Size (M, D)
    face = mesh['face']  # Size (Q, 3)

    faces_pos = pos[face]  # Size (Q, 3, D) -- position of the triangle vertices.

    """1. Choose a face of the mesh at random."""
    v1 = faces_pos[:, 1, :] - faces_pos[:, 0, :]  # Size (Q, D)
    v2 = faces_pos[:, 2, :] - faces_pos[:, 0, :]  # Size (Q, D)
    h = v2 - (v1 * v2).sum(dim=-1, keepdim=True) * v1 / torch.norm(v1, dim=-1, keepdim=True) ** 2  # Size (Q, D)

    face_areas = (torch.norm(h, dim=-1) * torch.norm(v1, dim=-1)) / 2  # Size (Q,)
    face_probabilities = face_areas / face_areas.sum()  # Size (Q,)

    sampled_face_indices = torch.multinomial(face_probabilities, num_samples=N, replacement=True)  # Size (N,)
    sampled_face_pos = faces_pos[sampled_face_indices]  # Size (N, 3, 3)

    """2. Sample uniformly at random on a 2D triangle."""
    sampled_flat_triangle = torch.rand(size=(N, 2))  # Size (N, 2)

    flip_pos = (sampled_flat_triangle[:, 0] + sampled_flat_triangle[:, 1]) > 1
    sampled_flat_triangle[flip_pos] = 1 - sampled_flat_triangle[flip_pos]

    """3. Map the 2D samples on their 3D faces."""
    points = sampled_face_pos[:, 2, :] + sampled_flat_triangle[:, 0, None] * (
            sampled_face_pos[:, 0, :] - sampled_face_pos[:, 2, :]) + sampled_flat_triangle[:, 1, None] * (
                     sampled_face_pos[:, 1, :] - sampled_face_pos[:, 2, :])  # Size (N, 3)

    return points


def decimate_mesh(mesh, n_target_points):
    """
    Simplify a mesh by reducing the number of vertices.

    Also computes a mapping from simplified vertices to original vertices by finding the nearest neighbor in the
    original mesh for each simplified vertex.

    Args:
        mesh: Dictionary containing:
            - 'pos': Original vertex positions, shape (M, D)
            - 'face': Original triangle faces, shape (Q, 3)
        n_target_points: Target number of vertices in simplified mesh

    Returns:
        torch.Tensor: Indices mapping simplified vertices to original vertices, shape (n_target_points,). For each
                      vertex in the simplified mesh, gives the index of its nearest vertex in the original.
        dict: Simplified mesh containing:
            - 'pos': Simplified vertex positions, shape (n_target_points, D)
            - 'face': Simplified triangle faces, shape (Q_reduced, 3)

    """
    pos = mesh['pos']  # Size (M, D)
    face = mesh['face']  # Size (Q, 3)

    pos_r, face_r, collapses = fast_simplification.simplify(pos, face, 1 - (n_target_points / pos.shape[0]),
                                                            return_collapses=True, )
    pos_r = torch.tensor(pos_r, dtype=torch.float32, device=pos.device)
    face_r = torch.tensor(face_r, dtype=torch.int64, device=face.device)

    X_i = LazyTensor(pos_r[:, None, :])
    X_j = LazyTensor(pos[None, :, :])

    D_ij = ((X_i - X_j) ** 2).sum(-1).sqrt()
    sub_inds = D_ij.argmin(dim=1).squeeze()

    mesh_r = {'pos': pos_r, 'face': face_r}
    return sub_inds, mesh_r
