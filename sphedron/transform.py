"""
Author: Ayoub Ghriss, dev@ayghri.com

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation

from .helpers import get_rotation_matrices as _get_rotations


def rotate_senders_by_receivers(
    receivers_xyz: NDArray,
    senders_xyz: NDArray,
    zero_latitude: bool = False,
    zero_longitude: bool = False,
):
    """
    Apply rotation that zeroes out receivers' latitude and/or longitude to
    Cartesian senders' coordinates.

    Args:
        receivers_xyz: An NDArray representing the coordinates of the
            receivers in Cartesian format, shape (N,3)
        senders_xyz: An NDArray representing the coordinates of the
            senders in Cartesian format, shape (N,3)
        zero_latitude: A boolean flag indicating whether to zero out the
            latitude of the receivers. Defaults to False.
        zero_longitude: A boolean flag indicating whether to zero out the
            longitude of the receivers. Defaults to False.

    Returns:
        NDArray for shape (N,3) of the senders' rotated coordinates
    """
    references_thetaphi = xyz_to_thetaphi(receivers_xyz)
    rotation_matrices = _get_rotations(
        references_thetaphi,
        zero_latitude=zero_latitude,
        zero_longitude=zero_longitude,
    )
    # faster than expand_dims or matmul
    return np.einsum("nij, nj-> ni", rotation_matrices, senders_xyz)


def sender2receiver_edge_coords(
    sender_nodes: NDArray,
    receiver_nodes: NDArray,
    sender2receiver_edges: NDArray,
    zero_latitude: bool = True,
    zero_longitude: bool = True,
):
    """
    Rotates the nodes so that the receivers have 0 latitude and longitude
    and computes the difference between the receivers and their senders coords

    Args:
        sender_nodes: An array of the nodes of the sender mesh, (N,3)
        receiver_nodes: An array of the nodes of receiver mesh, (M,3)
        sender2receiver_edges: An array representing edges from senders to
            receivers, (E, 2)
        zero_latitude: If True, adjusts the latitude of receivers to 0.
        zero_longitude: If True, adjusts the longitude of receivers to 0.

    Returns:
        An array of coordinates representing the differences between the
        receivers and their corresponding senders, shaped (E, 3)
    """
    senders_wrt_receivers = rotate_senders_by_receivers(
        receiver_nodes[sender2receiver_edges[:, 1]],
        sender_nodes[sender2receiver_edges[:, 0]],
        zero_latitude=zero_latitude,
        zero_longitude=zero_longitude,
    )
    receivers_wrt_receivers = rotate_senders_by_receivers(
        receiver_nodes[sender2receiver_edges[:, 1]],
        receiver_nodes[sender2receiver_edges[:, 1]],
        zero_latitude=zero_latitude,
        zero_longitude=zero_longitude,
    )

    return senders_wrt_receivers - receivers_wrt_receivers


def rotate_nodes(nodes_xyz: NDArray, axis: str, angle: float) -> NDArray:
    """
    Rotate the mesh nodes around a specified axis by a given angle.

    Args:
        nodes_xyz: Cartesian coordinates of the nodes, shape (N, 3)
        axis: axis around which to rotate the nodes, scipy's Euler rotation.
        angle: The angle (radian) by which to rotate the nodes.

    Returns:
        The rotated nodes of shape (num_nodes, 3).
    """
    assert len(axis) == 1
    rotation = Rotation.from_euler(seq=axis, angles=angle).as_matrix()
    nodes_xyz = np.dot(nodes_xyz, rotation.T)
    return nodes_xyz


def xyz_to_thetaphi(xyz: NDArray) -> NDArray:
    """Convert Cartesian coordinates to spherical"""
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    theta = np.arccos(xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.c_[theta, phi]


def thetaphi_to_xyz(thetaphi: NDArray):
    """Convert spherical coordinates on the sphere (r=1) to Cartesian"""
    return np.c_[
        np.cos(thetaphi[:, 1]) * np.sin(thetaphi[:, 0]),
        np.sin(thetaphi[:, 1]) * np.sin(thetaphi[:, 0]),
        np.cos(thetaphi[:, 0]),
    ]


def latlong_to_thetaphi(latlong: NDArray) -> NDArray:
    """
    Convert latitude-longitude to spherical (theta, phi)
    where theta is the inclination (angle from positive z-axis)
    and phi the azimuth (z-axis rotation).
    """
    return np.c_[np.deg2rad(90 - latlong[:, [0]]), np.deg2rad(latlong[:, [1]])]


def thetaphi_to_latlong(thetaphi: NDArray):
    """Convert spherical to latitude/longitude"""
    lats = 90 - np.rad2deg(thetaphi[:, 0])
    longs = np.rad2deg(thetaphi[:, 1])
    return np.c_[lats, longs]


def xyz_to_latlong(xyz: NDArray) -> NDArray:
    """Cartesian coordinates on the unit sphere to latitude,longitude"""
    return thetaphi_to_latlong(xyz_to_thetaphi(xyz))


def latlong_to_xyz(latlong: NDArray) -> NDArray:
    """latitude,longitude to Cartesian coordinates on the unit sphere"""
    return thetaphi_to_xyz(latlong_to_thetaphi(latlong))
