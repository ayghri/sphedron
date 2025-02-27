"""
Ayoub Ghriss, ayoub.ghriss@colorado.edu
Non-commercial use.
"""

from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


def faces_to_edges(faces: NDArray):
    """Convert faces to list of composing edges"""
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


def compute_angles_per_depth(max_depth=100):
    """Compute angles between nodes"""
    phi = (1 + np.sqrt(5)) / 2
    initial_nodes = np.array([[-1, -phi, 0], [1, -phi, 0]])
    initial_nodes = initial_nodes / np.linalg.norm(initial_nodes, axis=1, keepdims=True)
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
    """
    Calculate the angles between nodes based on the lengths of edges.

    Parameters:
    nodes (NDArray): An array of vertex coordinates.
    faces (NDArray): An array of face indices that define the connectivity of nodes.

    Returns:
    NDArray: An array of angles (in degrees) between nodes calculated from edge lengths.
    """
    edges_lengths = compute_edges_lenghts(nodes, faces)
    angles_between_nodes = 360 * np.arcsin(edges_lengths / 2) / np.pi
    return angles_between_nodes


def query_nearest(
    references_xyz: NDArray,
    nodes_xyz: NDArray,
    n_neighbors: int = -1,
    radius: float = -1.0,
) -> Tuple[NDArray, NDArray]:
    """
    Return the list of indices so that references_xyz[indices[j]] are the all n_neighbors
    or all points within radius r from nodes_xyz[j]

    Returns:
        Array of arrays, internal arrays are not necessarily of the same length when
        querying the radius
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


def rotate_nodes(nodes: NDArray, axis: str, angle: float):
    """
    Rotate the mesh
    Args:
        nodes: (np.array) shape (num_nodes, 3)
    Returns:
        nodes: (np.array) rotated nodes of shape (num_nodes, 3)
    """
    assert len(axis) == 1
    rotation = Rotation.from_euler(seq=axis, angles=angle).as_matrix()
    nodes = np.dot(nodes, rotation.T)
    return nodes


def xyz_to_thetaphi(xyz: NDArray) -> NDArray:
    """Convert catesian to spherical"""
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    theta = np.arccos(xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.c_[theta, phi]


def thetaphi_to_xyz(thetaphi):
    """Convert spherical coordinates on the sphere to cartesian coordinates"""
    return np.c_[
        np.cos(thetaphi[:, 1]) * np.sin(thetaphi[:, 0]),
        np.sin(thetaphi[:, 1]) * np.sin(thetaphi[:, 0]),
        np.cos(thetaphi[:, 0]),
    ]


def latlon_to_thetaphi(latlon: NDArray) -> NDArray:
    """
    Convert latlon to spherical (theta, phi) where theta is the inclination (angle
    from positive z-axis) and phi the azimuth (z-axis rotation)
    """
    return np.c_[np.deg2rad(90 - latlon[:, [0]]), np.deg2rad(latlon[:, [1]])]


def thetaphi_to_latlon(thetaphi: NDArray):
    """Convert spherical to latitude/longitude"""
    lats = 90 - np.rad2deg(thetaphi[:, 0])
    longs = np.rad2deg(thetaphi[:, 1])
    return np.c_[lats, longs]


def xyz_to_latlon(xyz: NDArray) -> NDArray:
    """
    Convert Cartesian coordinates on the unit sphere to longitude,latitude
    Args:
        xyz: Cartesian coordinates on the sphere, shape (K,3)
    Returns:
        latlon: array of shape (K,2)
    """
    return thetaphi_to_latlon(xyz_to_thetaphi(xyz))


def latlon_to_xyz(latlon: NDArray) -> NDArray:
    """
    Convert longitude,latitude to Cartesian coordinates on the unit sphere
    Args:
        latlon: array of shape (K,2)
    Returns:
        Cartesian coordinates on the sphere ,shape (K,3)
    """
    return thetaphi_to_xyz(latlon_to_thetaphi(latlon))
