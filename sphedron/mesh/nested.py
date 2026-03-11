from typing import List, Type
from numpy.typing import NDArray
import numpy as np

from .base import Mesh
from .refinables import Icosphere, Cubesphere, Octasphere


class NestedMeshes:
    """A manager for a hierarchy of meshes at increasing refinement levels.

    Composes multiple :class:`Mesh` objects.  Individual levels are
    accessible via indexing (``nested[i]``), and convenience properties
    aggregate nodes, edges, and faces across the hierarchy.

    Args:
        factors: List of refinement factors. Each entry refines the
            previous level by that factor. The cumulative product gives
            the effective depth at each level.
        refine_by_angle: Use angle-based (geodesic) interpolation during
            refinement.
        rotate: Rotate the base mesh using the class rotation parameters.

    Attributes:
        finest_mesh (Mesh): The mesh at the highest refinement level.
        nodes (NDArray): Nodes of the finest mesh, shape ``(N, 3)``.
        nodes_latlong (NDArray): Finest mesh nodes in ``(lat, lon)`` degrees.
        num_edges (int): Total edge count across all levels.
        num_faces (int): Total face count across all levels.
        num_nodes (int): Node count of the finest mesh.
        edges (NDArray): All edges from every level, concatenated.
        faces (NDArray): All faces from every level, concatenated.
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
            mesh = self._base_mesh_cls.from_graph(
                nodes,
                faces,
                refine_factor=factor,
                refine_by_angle=refine_by_angle,
            )
            self.meshes.append(mesh)
            nodes, faces = mesh._all_nodes, mesh._all_faces

    def __getitem__(self, level: int) -> Mesh:
        """Get the mesh at a specific refinement level."""
        return self.meshes[level]

    def __len__(self) -> int:
        """Return the number of refinement levels."""
        return len(self.meshes)

    def __repr__(self) -> str:
        return (
            f"NestedMeshes with {len(self.meshes)} levels\n"
            f"  factors: {self.metadata['factors']}\n"
            f"  #nodes (finest): {self.num_nodes}\n"
        )

    @property
    def finest_mesh(self) -> Mesh:
        """Returns the mesh at the highest refinement level."""
        return self.meshes[-1]

    def reset(self):
        """Resets the node masks on all meshes in the hierarchy."""
        for mesh in self.meshes:
            mesh.reset()

    def mask_nodes(self, nodes_mask: NDArray[np.bool_]):
        """Apply a mask to nodes, propagating to all refinement levels.

        Applies the mask to the finest mesh and propagates the masking
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
        """Nodes of the finest (most refined) mesh."""
        return self.meshes[-1].nodes

    @property
    def nodes_latlong(self):
        """Nodes of the finest mesh in latitude/longitude format (degrees)."""
        return self.meshes[-1].nodes_latlong

    @property
    def num_edges(self):
        """Total number of edges across all refinement levels."""
        return sum(mesh.num_edges for mesh in self.meshes)

    @property
    def num_faces(self):
        """Total number of faces across all refinement levels."""
        return sum(mesh.num_faces for mesh in self.meshes)

    @property
    def num_nodes(self):
        """Number of nodes in the finest mesh."""
        return self.meshes[-1].num_nodes

    @property
    def edges(self) -> NDArray[np.int_]:
        """All edges from every refinement level, concatenated."""
        return np.concatenate([mesh.edges for mesh in self.meshes], axis=0)

    @property
    def faces(self) -> NDArray[np.int_]:
        """All faces from every refinement level, concatenated."""
        return np.concatenate([mesh.faces for mesh in self.meshes], axis=0)

    def build_trimesh(self):
        """Build a Trimesh from the finest mesh."""
        return self.finest_mesh.build_trimesh()

    def query_edges_from_faces(self, receiver_mesh) -> NDArray:
        """Delegate face-based edge query to the finest mesh."""
        return self.finest_mesh.query_edges_from_faces(receiver_mesh)

    def query_edges_from_radius(self, receiver_mesh, radius: float) -> NDArray:
        """Delegate radius-based edge query to the finest mesh."""
        return self.finest_mesh.query_edges_from_radius(receiver_mesh, radius)

    def query_edges_from_neighbors(self, receiver_mesh, n_neighbors) -> NDArray:
        """Delegate neighbor-based edge query to the finest mesh."""
        return self.finest_mesh.query_edges_from_neighbors(
            receiver_mesh, n_neighbors
        )


class NestedCubespheres(NestedMeshes):
    """Nested cubespheres, where self.mesh[i+1] is a refined self.meshes[i]."""

    _base_mesh_cls = Cubesphere


class NestedOctaspheres(NestedMeshes):
    """Nested octaspheres, where self.mesh[i+1] is a refined self.meshes[i]."""

    _base_mesh_cls = Octasphere


class NestedIcospheres(NestedMeshes):
    """Nested icospheres, where each level is a refined version of the previous."""

    _base_mesh_cls = Icosphere
