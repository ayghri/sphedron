"""
License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree  # type: ignore


def faces_to_edges(faces: NDArray) -> NDArray:
    """Convert faces to an array of composing edges.

    Each face is defined by K node indices. This function returns all edges
    by pairing each node index with its successor in the face, wrapping
    around to the first node at the end.

    Args:
        faces (numpy.ndarray[M, K]): Array of faces, shape (M, K), where each
            row lists K node indices defining a single face.

    Returns:
        numpy.ndarray[N, 2]: Array of edges with shape (M*K, 2), where each
        row is a pair of node indices forming an edge.

    Example:
        >>> import numpy as np
        >>> faces = np.array([[0, 1, 2], [2, 3, 0]])
        >>> edges = faces_to_edges(faces)
        >>> edges
        array([[0, 1],
               [1, 2],
               [2, 0],
               [2, 3],
               [3, 0],
               [0, 2]])
    """
    # faces shape (M, K)
    num_edges_per_face = faces.shape[1]
    return np.concatenate(
        [
            faces[:, [i, (i + 1) % num_edges_per_face]]
            for i in range(num_edges_per_face)
        ],
        axis=0,
    )


def query_nearest(
    references_xyz: NDArray,
    nodes_xyz: NDArray,
    n_neighbors: int,
) -> Tuple[NDArray, NDArray]:
    """Find the nearest neighbors for each query point among reference points.

    Args:
        references_xyz (numpy.ndarray[N, 3]): Reference points in 3D space.
        nodes_xyz (numpy.ndarray[M, 3]): Query points in 3D space.
        n_neighbors (int): Number of nearest neighbors to return.

    Returns:
        Tuple[numpy.ndarray[M, n_neighbors], numpy.ndarray[M, n_neighbors]]:
            A pair `(distances, indices)` where `distances[i, j]` is the
            distance from `nodes_xyz[i]` to its j-th nearest neighbor among
            `references_xyz`, and `indices[i, j]` is the index of that neighbor.

    Example:
        >>> import numpy as np
        >>> refs = np.random.rand(10, 3)
        >>> nodes = np.random.rand(5, 3)
        >>> dists, idxs = query_nearest(refs, nodes, n_neighbors=3)
    """
    distances, indices = cKDTree(references_xyz).query(
        x=nodes_xyz, k=n_neighbors, workers=-1
    )
    return distances, indices


def query_radius(
    references_xyz: NDArray,
    nodes_xyz: NDArray,
    radius: float,
) -> NDArray:
    """Find all neighbors within a given radius for each query point.

    Args:
        references_xyz (numpy.ndarray[N, 3]): Reference points in 3D space.
        nodes_xyz (numpy.ndarray[M, 3]): Query points in 3D space.
        radius (float): Radius within which to search for neighbors.

    Returns:
        List[numpy.ndarray]: A list of length M; each entry is an array of
        indices of `references_xyz` that lie within `radius` of the
        corresponding `nodes_xyz` point.

    Example:
        >>> import numpy as np
        >>> refs = np.random.rand(10, 3)
        >>> nodes = np.random.rand(5, 3)
        >>> neighbors = query_radius(refs, nodes, radius=0.2)
    """
    indices = cKDTree(references_xyz).query_ball_point(
        x=nodes_xyz, r=radius, workers=-1
    )
    return indices


def form_edges(
    sender_groups: NDArray, receiver_indices: NDArray | List
) -> NDArray:
    """Connect sender nodes to receiver nodes and return the edge list.

    Args:
        sender_groups (numpy.ndarray or list): 1D or 2D array of sender node
            indices. Each entry can be a scalar or a list of indices.
        receiver_indices (numpy.ndarray or list): 1D array of receiver node
            indices, same length as `sender_groups`.

    Returns:
        numpy.ndarray[N, 2]: Array of edges where each row is
        `(sender_index, receiver_index)`.

    Example:
        >>> senders = [[0, 1], 2]
        >>> receivers = [3, 4]
        >>> edges = form_edges(senders, receivers)
        >>> print(edges)
        [[0, 3], [1, 3], [2, 4]]
    """
    edges = []
    for s_group, r_idx in zip(sender_groups, receiver_indices):
        if np.isscalar(s_group):
            s_group = [s_group]
        for s_idx in s_group:
            edges.append((s_idx, r_idx))
    return np.array(edges)


def get_rotation_matrices(
    references_thetaphi: NDArray,
    zero_latitude: bool,
    zero_longitude: bool,
):
    """Compute rotation matrices to align spherical coords to zero lat/long.

    Args:
        references_thetaphi (numpy.ndarray[N, 2]): Array of
            `(theta, phi)` angles in radians.
        zero_latitude (bool): If True, rotate so that `theta` → 0.
        zero_longitude (bool): If True, rotate so that `phi` → 0.

    Returns:
        numpy.ndarray[N, 3, 3]: Rotation matrix for each reference angle,
        suitable for `Rotation.from_euler(...).as_matrix()`.

    Raises:
        ValueError: If neither `zero_latitude` nor `zero_longitude` is enabled.

    Example:
        >>> import numpy as np
        >>> thph = np.array([[np.pi/4, np.pi/2]])
        >>> mats = get_rotation_matrices(thph, True, False)
        >>> mats.shape
        (1, 3, 3)
    """
    if not (zero_latitude or zero_longitude):
        raise ValueError(
            "At least one of zero_latitude or zero_longitude must be True."
        )
    azimuthal_rotation = -references_thetaphi[:, 1]

    if zero_latitude:
        polar_rotation = np.pi / 2 - references_thetaphi[:, 0]
        if zero_longitude:
            # first rotate on z, then on the new rotated y (not the absolute Y)
            return Rotation.from_euler(
                "zy",
                np.stack([azimuthal_rotation, polar_rotation], axis=1),
            ).as_matrix()
        # rotate on z, then on the new rotated, then undo z
        return Rotation.from_euler(
            "zyz",
            np.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation],
                axis=1,
            ),
        ).as_matrix()
    # reaching here -> zero_longitude only
    return Rotation.from_euler("z", azimuthal_rotation).as_matrix()


def compute_angles_per_depth(max_depth: int = 100) -> NDArray:
    """Compute angular separation between nodes as depth increases.

    Args:
        max_depth (int): Number of depth steps to compute (>=1).

    Returns:
        numpy.ndarray[max_depth]: Array of angles in degrees for each depth.

    Example:
        >>> angles = compute_angles_per_depth(5)
        >>> len(angles)
        5
    """
    phi = (1 + np.sqrt(5)) / 2
    initial_nodes = np.array([[-1, -phi, 0], [1, -phi, 0]])
    initial_nodes /= np.linalg.norm(initial_nodes, axis=1, keepdims=True)
    angles = []
    for d in range(1, max_depth + 1):
        left = initial_nodes[0]
        right = left + (initial_nodes[1] - initial_nodes[0]) / d
        nodes = np.stack([left, right], axis=0)
        nodes /= np.linalg.norm(nodes, axis=1, keepdims=True)
        angles.append(np.degrees(np.arccos(np.inner(nodes[0], nodes[1]))))
    return np.array(angles)


def compute_edges_lenghts(
    nodes: NDArray,
    edges: NDArray,
) -> NDArray:
    """Compute the lengths of edges given node coordinates.

    Args:
        nodes (numpy.ndarray[K, 3]): Array of node coordinates.
        edges (numpy.ndarray[E, 2]): Array of edges as index pairs.

    Returns:
        numpy.ndarray[E]: Euclidean lengths of each edge.

    Example:
        >>> import numpy as np
        >>> nodes = np.array([[0, 0, 0], [1, 0, 0]])
        >>> edges = np.array([[0, 1]])
        >>> lengths = compute_edges_lenghts(nodes, edges)
        >>> lengths
        array([1.])
    """
    edges_nodes = nodes[edges]  # shape (E, 2, 3)
    diffs = edges_nodes[:, 1] - edges_nodes[:, 0]
    return np.linalg.norm(diffs, axis=-1)


def compute_edges_angles(
    nodes: NDArray,
    faces: NDArray,
) -> NDArray:
    """Calculate node‐to‐node angles from edge lengths.

    Args:
        nodes (numpy.ndarray[K, 3]): Coordinates of nodes.
        faces (numpy.ndarray[M, 2]): Array defining edges as pairs of indices.

    Returns:
        numpy.ndarray[M]: Angles in degrees for each edge.

    Example:
        >>> import numpy as np
        >>> nodes = np.array([[0, 0, 1], [0, 1, 0]])
        >>> faces = np.array([[0, 1]])
        >>> angles = compute_edges_angles(nodes, faces)
        >>> angles
        array([90.])
    """
    lengths = compute_edges_lenghts(nodes, faces)
    return 360 * np.arcsin(lengths / 2) / np.pi
