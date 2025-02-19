"""
Ayoub Ghriss, ayoub.ghriss@colorado.edu
Non-commercial use.
"""

from typing import Callable, Dict, Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.spatial.transform import Rotation
from .mesh import Mesh
from .utils import find_nearest_nodes
from .utils import xyz_to_thetaphi


class MeshTransfer:
    """A class to facilitate the transfer of values between meshes

    Parameters
    ----------
    meshes : Dict[str, Mesh]
        A dictionary containing the longitude and latitude arrays for each mesh.
    n_neighbors: int, default=5
        Number of nearest neighbors to consider for value transfer.

    Attributes
    ----------
    nearest_neighbors :
    meshes :
    n_neighbors :
    meshes_latlons: Dict[str, NDArray], A dictionary mapping mesh names
        to their longitude and latitude coordinates.
    nearest_neighbors: Dict[Tuple[str, str], NDArray],
        A cache for storing nearest neighbor indices for mesh pairs.
    """

    def __init__(self, meshes: Dict[str, Mesh], n_neighbors: int = 5) -> None:
        """ """
        self.nearest_neighbors: Dict[Tuple[str, str], Tuple[NDArray, NDArray]] = {}
        self.meshes = meshes
        self.n_neighbors = n_neighbors

    def add_mesh(self, mesh_name, mesh: Mesh, replace=False):
        if mesh_name in self.meshes and not replace:
            raise AssertionError(
                f"Mesh {mesh} already exists. Use replace=True to overwrite"
            )
        else:
            self.meshes[mesh_name] = mesh

    def transfer(
        self,
        source_vals: NDArray,
        source_mesh: str,
        target_mesh: str,
        average: bool = True,
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
        assert (
            source_vals.shape[0] == self.meshes[source_mesh].num_nodes
        ), "sourcevalues and mesh do not have the number of nodes"
        nearest_vals = source_vals[self.get_neighbors(source_mesh, target_mesh)]
        if average:
            nearest_vals = nearest_vals.mean(1)
        return nearest_vals

    def weighted_transfer(
        self,
        source_vals: NDArray,
        source_mesh: str,
        target_mesh: str,
        weight_func: Callable | None = None,
    ) -> NDArray:
        """
        Transfers values from the source mesh to the target mesh using nearest neighbor interpolation.

        Parameters:
            source_vals (NDArray): The values to be transferred from the source mesh.
            source_mesh (str): The name of the source mesh.
            target_mesh (str): The name of the target mesh.
            #TODO: add weight_func

        Returns:
            NDArray: The transferred values for the target mesh.
        """
        assert (
            source_vals.shape[0] == self.meshes[source_mesh].num_nodes
        ), "sourcevalues and mesh do not have the number of nodes"
        nearest_idx, nearest_dist = self.get_neighbors(
            source_mesh,
            target_mesh,
            get_distances=True,
        )
        nearest_vals = source_vals[nearest_idx]
        if weight_func is None:
            weight_func = lambda x: np.ones_like(x)
        weights = weight_func(nearest_dist)
        weights = weights / weights.sum(axis=1, keepdims=True)

        return (nearest_vals * weights).sum(axis=1)

    def get_neighbors(self, source_mesh, target_mesh, get_distances=False):
        if (source_mesh, target_mesh) not in self.nearest_neighbors:
            self.nearest_neighbors[(source_mesh, target_mesh)] = find_nearest_nodes(
                self.meshes[source_mesh].nodes,
                self.meshes[target_mesh].nodes,
                n_neighbors=self.n_neighbors,
            )
        if not get_distances:
            return self.nearest_neighbors[(source_mesh, target_mesh)][0]

        return self.nearest_neighbors[(source_mesh, target_mesh)]


def rotate_nodes_by_references(
    references_xyz: NDArray,
    nodes_xyz: NDArray,
    zero_latitude: bool,
    zero_longitude: bool,
):
    """
    Adapted from:
        Repo: https://github.com/google-deepmind/graphcast
        path: graphcast/model_utils.py
    Compute senders' cartesian coordinates in a reference where receivers lat/lon
    are set to 0 (theta=pi/2, phi=0)
    """
    if not (zero_latitude or zero_longitude):
        raise ValueError("zero longitude and/or latitude should be set!")
    references_thetaphi = xyz_to_thetaphi(references_xyz)

    azimuthal_rotation = -references_thetaphi[:, 1]

    if zero_longitude:
        if zero_latitude:
            # first rotate on z, then on the new rotated y (not the absolute Y)
            polar_rotation = np.pi / 2 - references_thetaphi[:, 0]
            rotation_matrices = Rotation.from_euler(
                "zy",
                np.stack(
                    [azimuthal_rotation, polar_rotation],
                    axis=1,
                ),
            ).as_matrix()
        else:
            # rotate on z
            rotation_matrices = Rotation.from_euler("z", azimuthal_rotation).as_matrix()

    else:
        # rotate on z, then on the new rotated, then undo z
        polar_rotation = np.pi / 2 - references_thetaphi[:, 0]
        rotation_matrices = Rotation.from_euler(
            "zyz",
            np.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation],
                axis=1,
            ),
        ).as_matrix()
    # this is faster than [:,None...] or matmul(mat,xyz)
    return np.einsum("nij, nj-> ni", rotation_matrices, nodes_xyz)
