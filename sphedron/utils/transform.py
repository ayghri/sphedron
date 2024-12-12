import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


def xyz_to_thetaphi(xyz: npt.NDArray) -> npt.NDArray:
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    theta = np.arccos(xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.c_[theta, phi]


def thetaphi_to_xyz(thetaphi):
    """
    Convert spherical coordinates on the sphere to cartesian coordinates
    """
    return np.c_[
        np.cos(thetaphi[:, 1]) * np.sin(thetaphi[:, 0]),
        np.sin(thetaphi[:, 1]) * np.sin(thetaphi[:, 0]),
        np.cos(thetaphi[:, 0]),
    ]


def latlon_to_thetaphi(latlon: npt.NDArray) -> npt.NDArray:
    """
    Convert latlon to spherical (theta, phi) where theta is the inclination (angle
    from positive z-axis) and phi the azimuth (z-axis rotation)
    """
    return np.c_[np.deg2rad(90 - latlon[:, [0]]), np.deg2rad(latlon[:, [1]])]


def thetaphi_to_latlon(thetaphi: npt.NDArray):
    lats = 90 - np.rad2deg(thetaphi[:, 0])
    longs = np.rad2deg(thetaphi[:, 1])
    return np.c_[lats, longs]


def xyz_to_latlon(xyz: npt.NDArray) -> npt.NDArray:
    """
    Convert Cartesian coordinates on the unit sphere to longitude,latitude
    Args:
        xyz: Cartesian coordinates on the sphere, shape (K,3)
    Returns:
        latlon: array of shape (K,2)
    """
    return thetaphi_to_latlon(xyz_to_thetaphi(xyz))


def latlon_to_xyz(latlon: npt.NDArray) -> npt.NDArray:
    """
    Convert longitude,latitude to Cartesian coordinates on the unit sphere
    Args:
        latlon: array of shape (K,2)
    Returns:
        Cartesian coordinates on the sphere ,shape (K,3)
    """
    return thetaphi_to_xyz(latlon_to_thetaphi(latlon))


def senders_relative_xyz(
    receivers_thetaphi: npt.NDArray,
    receivers_xyz: npt.NDArray,
    senders_xyz,
    zero_latitude,
    zero_longitude: bool,
):
    """
    Adapted from:
        Repo: https://github.com/google-deepmind/graphcast
        path: graphcast/model_utils.py
    Compute relative position of senders to receivers (senders_xyz-receivers_xyz)
    in rotated reference in which senders are at zero lat/lon
    """
    assert zero_latitude or zero_longitude

    if zero_longitude:
        if zero_latitude:
            rot = Rotation.from_euler(
                "zy",
                np.stack(
                    [-receivers_thetaphi[:, 1], receivers_thetaphi[:, 0] + np.pi / 2],
                    axis=1,
                ),
            ).as_matrix()
        else:
            rot = Rotation.from_euler("z", -receivers_thetaphi[:, 1]).as_matrix()

    else:
        rot = Rotation.from_euler(
            "zyz",
            np.stack(
                [
                    -receivers_thetaphi[:, 1],
                    -receivers_thetaphi[:, 0] + np.pi / 2,
                    +receivers_thetaphi[:, 1],
                ],
                axis=1,
            ),
        ).as_matrix()

    # rot has shape (N, 3, 3)
    # senders shape (N, 3)

    return (rot * senders_xyz[:, None]).sum(axis=-11) - receivers_xyz
