"""
Ayoub Ghriss, dev@ayghri.com
Non-commercial use.
"""

from typing import Callable, Dict, Tuple
import numpy.typing as ntp
from .utils.mesh import change_grid
from .mesh import Mesh


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
    meshes_latlons: Dict[str, ntp.NDArray], A dictionary mapping mesh names
        to their longitude and latitude coordinates.
    nearest_neighbors: Dict[Tuple[str, str], ntp.NDArray],
        A cache for storing nearest neighbor indices for mesh pairs.
    """

    def __init__(self, meshes: Dict[str, Mesh], n_neighbors: int = 5) -> None:
        """ """
        self.nearest_neighbors: Dict[
            Tuple[str, str], Tuple[ntp.NDArray, ntp.NDArray]
        ] = {}
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
        source_vals: ntp.NDArray,
        source_mesh: str,
        target_mesh: str,
        average: bool = True,
    ) -> ntp.NDArray:
        """
        Transfers values from the source mesh to the target mesh using nearest neighbor interpolation.

        Parameters:
            source_vals (ntp.NDArray): The values to be transferred from the source mesh.
            source_mesh (str): The name of the source mesh.
            target_mesh (str): The name of the target mesh.
            average (bool, optional): If True, averages the values of the nearest neighbors. Defaults to True.

        Returns:
            ntp.NDArray: The transferred values for the target mesh.
        """
        assert (
            source_vals.shape[0] == self.meshes[source_mesh].num_vertices
        ), "sourcevalues and mesh do not have the number of vertices"
        nearest_vals = source_vals[self.get_neighbors(source_mesh, target_mesh)]
        if average:
            nearest_vals = nearest_vals.mean(1)
        return nearest_vals

    def weighted_transfer(
        self,
        source_vals: ntp.NDArray,
        source_mesh: str,
        target_mesh: str,
        weight_func: Callable,
    ) -> ntp.NDArray:
        """
        Transfers values from the source mesh to the target mesh using nearest neighbor interpolation.

        Parameters:
            source_vals (ntp.NDArray): The values to be transferred from the source mesh.
            source_mesh (str): The name of the source mesh.
            target_mesh (str): The name of the target mesh.
            #TODO: add weight_func

        Returns:
            ntp.NDArray: The transferred values for the target mesh.
        """
        assert (
            source_vals.shape[0] == self.meshes[source_mesh].num_vertices
        ), "sourcevalues and mesh do not have the number of vertices"
        nearest_idx, nearest_dist = self.get_neighbors(
            source_mesh,
            target_mesh,
            get_distances=True,
        )
        nearest_vals = source_vals[nearest_idx]
        weights = weight_func(nearest_dist)
        weights = weights / weights.sum(axis=1, keepdims=True)

        return (nearest_vals * weights).sum(axis=1)

    def get_neighbors(self, source_mesh, target_mesh, get_distances=False):
        if (source_mesh, target_mesh) not in self.nearest_neighbors:
            self.nearest_neighbors[(source_mesh, target_mesh)] = change_grid(
                self.meshes[source_mesh].vertices_latlon,
                self.meshes[target_mesh].vertices_latlon,
                n_neighbors=self.n_neighbors,
            )
        if not get_distances:
            return self.nearest_neighbors[(source_mesh, target_mesh)][0]

        return self.nearest_neighbors[(source_mesh, target_mesh)]
