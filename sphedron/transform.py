"""
Ayoub Ghriss, ayoub.ghriss@colorado.edu
Non-commercial use.
"""

from typing import Callable, Dict, Literal, Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation
from .mesh import Mesh
from .utils import query_nearest
from .utils import xyz_to_thetaphi


class MeshTransfer:
    """A class to facilitate the transfer of values between meshes"""

    def __init__(
        self,
        source_mesh: Mesh,
        target_mesh: Mesh,
        n_neighbors: int = 5,
    ) -> None:
        """ """
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.n_neighbors = n_neighbors
        self.nearest_neighbors = None

    def transfer(
        self,
        source_vals: NDArray,
        aggregation: Literal["mean", "sum", "max", "min"] = "mean",
        recompute: bool = False,
    ) -> NDArray:
        """
        Transfers values from the source mesh to the target mesh using nearest neighbor interpolation.

        Parameters:
            source_vals (NDArray): The values to be transferred from the source mesh.
            source_mesh (str): The name of the source mesh.
            target_mesh (str): The name of the target mesh.
            average (bool, optional): If True, averages the values of the nearest neighbors. Defaults to True.

        Returns:
            NDArray: The transferred values for the target mesh.
        """
        if source_vals.shape[0] != self.source_mesh.num_nodes:
            raise ValueError(
                f"source_values and mesh do not have the same number of nodes"
            )
        nearest_vals = source_vals[self.get_neighbors(recompute)]
        aggregated_values = None
        try:
            aggregated_values = getattr(np, "aggregation")(nearest_vals, axis=1)
            return aggregated_values
        except AttributeError:
            raise ValueError(
                f"aggregation {aggregation} is invalid, allowed : ['mean', 'sum', 'max', 'min']"
            )

    def weighted_transfer(
        self,
        source_vals: NDArray,
        weight_func: Callable | None = None,
        recompute: bool = False,
    ) -> NDArray:
        """
        Transfers values from the source mesh to the target mesh using nearest neighbor interpolation.

        Parameters:
            source_vals (NDArray): The values to be transferred from the source mesh.
            #TODO: add weight_func

        Returns:
            NDArray: The transferred values for the target mesh.
        """
        nearest_idx, nearest_dist = self.get_neighbors(recompute, get_distances=True)
        nearest_vals = source_vals[nearest_idx]
        if weight_func is None:
            weight_func = lambda x: np.ones_like(x)
        weights = weight_func(nearest_dist)
        weights = weights / weights.sum(axis=1, keepdims=True)

        return (nearest_vals * weights).sum(axis=1)

    def get_neighbors(self, recompute: bool, get_distances=False):
        if self.nearest_neighbors is None or recompute:
            self.nearest_neighbors = query_nearest(
                self.source_mesh.nodes,
                self.target_mesh.nodes,
                n_neighbors=self.n_neighbors,
            )
        if not get_distances:
            return self.nearest_neighbors[0]

        return self.nearest_neighbors


def get_rotation_matrices(references_thetaphi, zero_latitude, zero_longitude):
    """
    Adapted from:
        Repo: https://github.com/google-deepmind/graphcast
        path: graphcast/model_utils.py
    Compute rotation matrices so that the references (theta, phi) become (pi/2, 0)
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


def rotate_senders_by_receivers(
    receivers_xyz: NDArray,
    senders_xyz: NDArray,
    zero_latitude: bool = False,
    zero_longitude: bool = False,
):
    """
    Apply rotation that zeroes out receivers' lat and/or lon to cartesian
    senders' coordinates.
    """
    references_thetaphi = xyz_to_thetaphi(receivers_xyz)
    rotation_matrices = get_rotation_matrices(
        references_thetaphi,
        zero_latitude=zero_latitude,
        zero_longitude=zero_longitude,
    )
    # faster than expand_dims or matmul
    return np.einsum("nij, nj-> ni", rotation_matrices, senders_xyz)


def sender2receiver_edge_coords(sender_nodes, receiver_nodes, sender2receiver_edges):

    # senders_xyz=sender_mesh.nodes[]
    # receivers_xyz= grid.nodes[mesh2g_edges[:,1]]
    # receiver_nodes = 0
    # sender_nodes = 0
    senders_wrt_receivers = rotate_senders_by_receivers(
        receiver_nodes, sender_nodes, zero_latitude=True, zero_longitude=True
    )
    receivers_wrt_receivers = rotate_senders_by_receivers(
        receiver_nodes, receiver_nodes, zero_latitude=True, zero_longitude=True
    )
