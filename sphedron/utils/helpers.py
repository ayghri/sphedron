"""
Author: Ayoub Ghriss, dev@ayghri.com
Date: 2024

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import List
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


def faces_to_edges(faces: NDArray) -> NDArray:
    """Convert faces to list of composing edges.

    Args:
        faces: array representing the faces, where each face is
               defined by a list of node indices, shape (N, K)

    Returns:
        Array of edges, shape (N*K, 2)
    """
    # faces shape (N, k) usually k = 3
    num_edges_per_face = faces.shape[1]
    # return the pairs face[i,(i+1) % N]
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
    radius: float = -1.0,
    n_neighbors: int = -1,
) -> NDArray:
    """
    Find the nearest neighbors for a set of nodes based on given reference
    points.

    This function returns the indices of the nearest reference neighbors
    for each point in `nodes_xyz`. The neighbors can be determined either by
    a specified number of neighbors (`n_neighbors`) or by a specified
    radius (`radius`). If both parameters are set, the function will
    prioritize the radius.

    Args:
        references_xyz: reference points, shape (N, 3)
        nodes_xyz: nodes for which to find nearest neighbors, shape (M, 3)
        radius: The radius to consider for finding the neighbors.
        n_neighbors: The number of nearest neighbors to return for each node.

    Returns:
        An array of arrays of indices of the nearest neighbors,
        of shape (M, n_neighbors) when n_neighbors is set
    """
    # either n_neighbors or radius should be used but not both
    # assert n_neighbors * radius < 0
    if radius > 0:
        indices = cKDTree(references_xyz).query_ball_point(
            x=nodes_xyz, r=radius, workers=-1
        )
    else:
        if n_neighbors < 0:
            raise ValueError(
                "Either radius or n_neighbors should be provided,"
                f"(n_neighbors, radius)=({n_neighbors}, {radius})  "
            )

        _, indices = cKDTree(references_xyz).query(
            x=nodes_xyz, k=n_neighbors, workers=-1
        )
    return indices


def connect_nodes(
    sender_groups: NDArray, receiver_indices: NDArray | List
) -> NDArray:
    """Connect sender nodes to receiver nodes and return the formed edges.

    This function takes groups of sender nodes and their corresponding receiver
    index, and creates an array of edges that represent the connections between
    the elements of the sender groups and receivers.

    Args:
        sender_groups: 1d or 2d array representing the indices of sender nodes.
        receiver_indices: 1d array of indices of the receiver nodes.

    Returns:
        (N,2) shaped array where each edge is (sender_index, receiver_index)
    """

    edges = []
    for senders_i, r_i in zip(sender_groups, receiver_indices):
        if len(senders_i) > 0:
            for s_i in senders_i:
                edges.append([s_i, r_i])
        else:
            edges.append([senders_i, r_i])
    return np.array(edges)


def get_rotation_matrices(
    references_thetaphi: NDArray,
    zero_latitude: bool,
    zero_longitude: bool,
):
    """Compute rotation matrices that align reference angles to zero latitude
    and/or longitude.

    This function calculates rotation matrices that transform the given
    reference angles (theta, phi). The parameters `zero_latitude` and
    `zero_longitude` determine whether the rotation should account for
    adjustments in latitude and longitude. Adapted from:
        Repo: https://github.com/google-deepmind/graphcast
        path: graphcast/model_utils.py

    Args:
        references_thetaphi: An array of reference angles in the form of
            (theta, phi), shape (N,2)
        zero_latitude: A boolean indicating if the rotation should adjust
            for zero latitude.
        zero_longitude: A boolean indicating if the rotation should adjust
            for zero longitude.

    Returns:
        NDArray: The computed rotation matrices, shape (N, 3, 3)
    """
    if not (zero_latitude or zero_longitude):
        raise ValueError(
            "at least on of zero longitude and zero_latitude should be enabled"
        )
    azimuthal_rotation = -references_thetaphi[:, 1]

    if zero_latitude:
        polar_rotation = np.pi / 2 - references_thetaphi[:, 0]
        if zero_longitude:
            # first rotate on z, then on the new rotated y (not the absolute Y)
            return Rotation.from_euler(
                "zy",
                np.stack(
                    [azimuthal_rotation, polar_rotation],
                    axis=1,
                ),
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


def compute_angles_per_depth(max_depth=100):
    """Compute angles between nodes"""
    phi = (1 + np.sqrt(5)) / 2
    initial_nodes = np.array([[-1, -phi, 0], [1, -phi, 0]])
    initial_nodes = initial_nodes / np.linalg.norm(
        initial_nodes, axis=1, keepdims=True
    )
    angles = []
    for d in range(1, max_depth + 1):
        left_vertex = initial_nodes[0]
        right_vertex = left_vertex + (initial_nodes[1] - initial_nodes[0]) / d
        nodes = np.array([left_vertex, right_vertex])
        nodes = nodes / np.linalg.norm(nodes, axis=1, keepdims=True)
        angles.append(np.arccos(np.inner(nodes[0], nodes[1])) / np.pi * 180)
    return np.array(angles)


def compute_edges_lenghts(
    nodes: NDArray,
    edges: NDArray[np.int_],
) -> NDArray:
    """Given the nodes and the edges, compute the length of each edge

    Args:
        nodes (numpy array): shape (K,3)
        edges (numpy array): shape (2,E)
    Returns:
        Lengths of the edges, shape (E,)
    """
    # edges: shape (2,E)
    edges_nodes = nodes[edges]  # shape (2, E, 3)
    edges_diff = edges_nodes[1] - edges_nodes[0]
    edges_lengths = np.linalg.norm(edges_diff, axis=-1)  # shape (E,)
    return edges_lengths


def compute_edges_angles(
    nodes: NDArray,
    faces: NDArray,
) -> NDArray:
    """Calculate the angles between nodes based on the lengths of edges.

    This function computes the angles formed at each node by the edges
    connecting it to its neighboring nodes. The angles are derived from
    the lengths of the edges defined by the provided nodes and faces.

    Parameters:
        nodes: An array of vertex coordinates.
        faces: An array of face indices that define the connectivity of nodes.

    Returns:
        An array of angles (in degrees) between nodes calculated from
        edge lengths.
    """
    edges_lengths = compute_edges_lenghts(nodes, faces)
    angles_between_nodes = 360 * np.arcsin(edges_lengths / 2) / np.pi
    return angles_between_nodes
