"""
License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import List, Type
from numpy.typing import NDArray
import numpy as np

from .base import Mesh
from .refinables import Icosphere, Cubesphere, Octasphere


class NestedMeshes(Mesh):
    """
    A manager for a hierarchy of meshes, where each mesh is a refinement
    of the previous one.

    This class composes multiple Mesh objects rather than inheriting from Mesh,
    providing a clearer and more robust API.
    """

    _base_mesh_cls: Type[Mesh] = Mesh

    def __init__(
        self,
        factors: List[int],
        refine_by_angle: bool = False,
        rotate: bool = True,
    ):
        assert issubclass(self._base_mesh_cls, Mesh)
        assert np.all(np.array(factors) >= 1)

        self.metadata = {
            "factors": factors,
            "cumulative_factors": list(np.cumprod(factors)),
        }
        self.meshes: List[Mesh] = []

        # Create the hierarchy of meshes
        mesh0 = self._base_mesh_cls.from_base(refine_factor=1, rotate=rotate)
        nodes, faces = mesh0._all_nodes, mesh0._all_faces

        for factor in factors:
            # Create a new mesh by refining the previous one
            mesh = self._base_mesh_cls.from_graph(
                nodes,
                faces,
                refine_factor=factor,
                refine_by_angle=refine_by_angle,
            )
            self.meshes.append(mesh)
            nodes, faces = mesh._all_nodes, mesh._all_faces
        super().__init__(nodes, faces)

    def __getitem__(self, level: int) -> Mesh:
        """Get the mesh at a specific refinement level."""
        return self.meshes[level]

    def __len__(self) -> int:
        """Return the number of refinement levels."""
        return len(self.meshes)

    @property
    def finest_mesh(self) -> Mesh:
        """Returns the mesh at the highest refinement level."""
        return self.meshes[-1]

    def reset(self):
        """Resets the node masks on all meshes in the hierarchy."""
        for mesh in self.meshes:
            mesh.reset()

    def mask_nodes(self, nodes_mask: NDArray[np.bool_]):
        """
        Applies a mask to the finest mesh and propagates the masking
        effect to coarser meshes where applicable.
        """
        if nodes_mask.shape[0] != self.num_nodes:
            raise ValueError(
                f"Nodes mask should have num_nodes={self.num_nodes} entries"
            )

        # This simple masking assumes nodes are perfectly nested.
        for mesh in self.meshes:
            mesh.mask_nodes(nodes_mask[: mesh.num_nodes])

    @property
    def nodes(self):
        return self.meshes[-1].nodes

    @property
    def num_edges(self):
        return sum((mesh.num_edges for mesh in self.meshes))

    @property
    def num_faces(self):
        return sum((mesh.num_faces for mesh in self.meshes))

    @property
    def num_nodes(self):
        return self.meshes[-1].num_nodes

    @property
    def edges(self) -> NDArray[np.int_]:
        return np.concatenate([mesh.edges for mesh in self.meshes], axis=0)

    @property
    def faces(self) -> NDArray[np.int_]:
        return np.concatenate([mesh.faces for mesh in self.meshes], axis=0)


class NestedCubespheres(NestedMeshes):
    """Nested cubespheres, where self.mesh[i+1] is a refined self.meshes[i]."""

    _base_mesh_cls = Cubesphere


class NestedOctaspheres(NestedMeshes):
    """Nested octaspheres, where self.mesh[i+1] is a refined self.meshes[i]."""

    _base_mesh_cls = Octasphere


class NestedIcospheres(NestedMeshes):
    _base_mesh_cls = Icosphere
