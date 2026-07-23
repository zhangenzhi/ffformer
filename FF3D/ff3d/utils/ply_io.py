"""PLY file I/O utilities."""
import os
import numpy as np


def save_ply_predictions(points, semantic_pred, instance_pred, scores, filename,
                         semantic_gt=None, instance_gt=None):
    """Save predictions as binary PLY with scalar fields."""
    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    N = points.shape[0]
    has_gt = semantic_gt is not None and instance_gt is not None

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('semantic_pred', 'i4'), ('instance_pred', 'i4'), ('score', 'f4')]
    if has_gt:
        dtype += [('semantic_gt', 'i4'), ('instance_gt', 'i4')]

    vertex = np.empty(N, dtype=dtype)
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['semantic_pred'] = semantic_pred
    vertex['instance_pred'] = instance_pred
    vertex['score'] = scores
    if has_gt:
        vertex['semantic_gt'] = semantic_gt
        vertex['instance_gt'] = instance_gt

    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(filename)


def save_ply_rgb(points, r, g, b, filename):
    """Save point cloud with RGB colors as binary PLY."""
    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    N = points.shape[0]
    dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex = np.empty(N, dtype=dtype)
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['red'] = r
    vertex['green'] = g
    vertex['blue'] = b

    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(filename)
