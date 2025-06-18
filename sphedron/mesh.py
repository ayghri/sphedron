"""
Author: Ayoub Ghriss, dev@ayghri.com

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import Tuple, Optional, Type, List
from numpy.typing import NDArray
import numpy as np
import trimesh


from .refine import refine_triangles
from .refine import refine_rectrangles
from .transform import xyz_to_latlong
from .transform import rotate_nodes
from .transform import latlong_to_xyz
from .helpers import faces_to_edges
from .helpers import query_nearest
from .helpers import query_radius
from .helpers import form_edges


class Mesh:
    """
    A flexible base class for a mesh on a sphere.
    Provides core functionality like node/face storage, masking, and
    cached properties. It does not dictate how a mesh is created,
    allowing for maximum flexibility in subclasses.
    """

    rotation_axis = "y"
    rotation_angle = 0

    def __init__(self, nodes: Optional[NDArray], faces: Optional[NDArray]):
        if nodes is None or faces is None:
            return

        self.metadata = {}
        self._all_nodes = nodes
        self._all_faces = faces
        self._cached_properties = {}
        self.reset()

    @classmethod
    def from_base(
        cls,
        refine_factor: int = 1,
        rotate: bool = True,
        refine_by_angle: bool = False,
    ):
        """Create an instance of the class from the base.

        This method generates nodes and faces from the base configuration
        returned by cls._base. If the `rotate` parameter is set to True, nodes
        will be rotated according to the class's rotation_{axis,angle}.
        The resulting nodes and faces are then passed to the `from_graph`
        method to create an instance of the class.

        Args:
            refine_factor: The factor by which to refine the mesh.
            rotate: A boolean indicating whether to rotate the nodes.
            refine_by_angle: A boolean indicating whether to refine by
                angle.

        Returns:
            An instance of the class created from the generated nodes
            and faces.
        """
        nodes, faces = cls._base()
        if rotate:
            nodes = rotate_nodes(
                nodes,
                axis=cls.rotation_axis,
                angle=cls.rotation_angle,
            )

        return cls.from_graph(
            nodes,
            faces,
            refine_factor=refine_factor,
            refine_by_angle=refine_by_angle,
        )

    @classmethod
    def from_graph(
        cls,
        nodes,
        faces,
        refine_factor: int = 1,
        refine_by_angle: bool = False,
    ):
        """Creates a refined mesh from the given base nodes and faces.

        This method refines the input mesh defined by `base_nodes` and
        `base_faces` using the specified `refine_factor`. The refinement
        can be controlled by the `refine_by_angle` parameter, which, if
        set to True, will refine the mesh based on the angles formed between
        of the the nodes.

        cls._refine handles the refinement of the mesh, which should be
        implemented for different types of faces.

        Args:
            base_nodes: The initial set of nodes defining the mesh.
            base_faces: The initial set of faces defining the mesh.
            refine_factor: The factor by which to refine the mesh.
            refine_by_angle: A boolean flag indicating whether to refine
                the mesh based on angles. Defaults to False.

        Returns:
            An instance of the refined mesh.
        """
        nodes, faces = cls._refine(
            nodes,
            faces,
            refine_factor=refine_factor,
            refine_by_angle=refine_by_angle,
        )
        return cls(nodes=nodes, faces=faces)

    @staticmethod
    def _refine(
        nodes,
        faces,
        refine_factor,
        refine_by_angle=False,
        **kwargs,
    ) -> Tuple[NDArray, NDArray]:
        raise NotImplementedError

    @staticmethod
    def _base() -> Tuple[NDArray, NDArray]:
        raise NotImplementedError

    # def reset(self):
    #     """
    #     Resets the mesh to its initial state by reinitializing the nodes mask
    #     """
    #     self._nodes_to_keep = np.ones(self._all_nodes.shape[0], dtype=bool)

    def __repr__(self) -> str:
        return (
            f"Mesh has: #nodes: {self.num_nodes}\n"
            f"          #faces: {self.num_faces}\n"
            f"          #edges: {self.num_edges}\n"
            f"          #edges_unique: {self.edges_unique.shape[0]}\n"
            f"          metadata: {self.metadata}"
        )

    @property
    def _node_sorting(self):
        # simple trick that is equivalent to mapping each retained node to its
        # position among all the retained nodes
        return np.cumsum(self._nodes_to_keep) - 1

    @property
    def edges(self):
        """Get the edges of the mesh"""
        edges = faces_to_edges(self._all_faces)
        # discard edges connected to a masked node
        edges_to_keep = np.all(self._nodes_to_keep[edges], axis=1)
        return self._node_sorting[edges[edges_to_keep]]

    @property
    def edges_symmetric(self) -> NDArray[np.int_]:
        """Get the edges of the undirected mesh"""
        return np.r_[self.edges_unique, self.edges_unique[:, ::-1]]

    @property
    def edges_unique(self):
        """Get directed edges of the graph, from lower to higher index nodes"""
        return np.unique(np.sort(self.edges, axis=1), axis=0)

    @property
    def faces(self) -> NDArray[np.int_]:
        """Get faces of the mesh, can be rectangles or triangles or more"""
        retained_faces = np.all(self._nodes_to_keep[self._all_faces], axis=1)
        return self._node_sorting[self._all_faces[retained_faces]]

    @property
    def faces_partial(self) -> NDArray:
        """Get faces of the mesh including those with partially masked nodes"""
        retained_faces = np.any(self._nodes_to_keep[self._all_faces], axis=1)
        return self._node_sorting[self._all_faces[retained_faces]]

    @property
    def nodes(self):
        """Returns the nodes of mesh in cartesian coordinates"""
        return self._all_nodes[self._nodes_to_keep]

    @property
    def nodes_latlong(self):
        """Returns the nodes of mesh in latitude/longitude format"""
        return xyz_to_latlong(self.nodes)

    @property
    def num_edges(self):
        """Number of edges"""
        return self.edges.shape[0]

    @property
    def num_faces(self) -> int:
        """Number of faces"""
        retained_faces = np.all(self._nodes_to_keep[self._all_faces], axis=1)
        return int(np.sum(retained_faces))

    @property
    def num_nodes(self):
        """Number of nodes"""
        return np.sum(self._nodes_to_keep)

    @property
    def triangles(self):
        """Return triangles of the mesh, useful for trimesh construction"""
        return self.faces2triangles(self.faces)

    def triangle2face_index(self, triangle_idx):
        """Convert triangle index to face index"""
        raise NotImplementedError

    def face2triangle_index(self, face_idx):
        """Convert face index to triangle index"""
        raise NotImplementedError

    def faces2triangles(self, faces):
        """Convert face to their corresponding triangles"""
        raise NotImplementedError

    def mask_nodes(self, nodes_mask: NDArray[np.bool_]):
        """Mask the nodes associated with nodes_mask[i] == True.

        This method expects a boolean mask of size (num_nodes,).
        It will mask the specified nodes but will not unmask any nodes that
        have previously been masked.

        Args:
            nodes_mask: A boolean array indicating which nodes to mask.
                        The array should have a size equal to the number of
                        nodes in the graph.
        """

        if nodes_mask.shape[0] != self.num_nodes:
            raise ValueError(
                f"Nodes mask should have num_nodes={self.num_nodes} entries"
            )
        assert self.num_nodes == nodes_mask.shape[0]
        self._nodes_to_keep[self._nodes_to_keep] = ~nodes_mask

    def build_trimesh(self):
        """return Trimesh instance from the current nodes and faces"""

        mesh = trimesh.Trimesh(self.nodes, self.triangles)
        mesh.fix_normals()
        return mesh

    def query_edges_from_faces(self, receiver_mesh: "Mesh") -> NDArray:
        """Get edges connecting nodes of the nearest face to receiver nodes.

        This method retrieves the edges that connect the nodes of the
        nearest face in the current mesh to the nodes in the provided
        receiver mesh.

        Args:
            receiver_mesh: The mesh object that contains the receiver nodes.

        Returns:
            An NDArray containing the edges that connect the nodes of the
            nearest face to the receiver nodes, shape (N,2)
        """
        # sender_trimesh = self.build_trimesh()
        sender_trimesh = self.build_trimesh()
        # shape (N,3)
        receiver_nodes = receiver_mesh.nodes
        _, _, query_triangle_indices = trimesh.proximity.closest_point(
            sender_trimesh, receiver_nodes
        )
        # shape (N,face_length)
        nearest_faces = self.faces[self.triangle2face_index(query_triangle_indices)]

        receiver_indices = np.tile(
            np.arange(receiver_nodes.shape[0])[:, None],
            (1, nearest_faces.shape[1]),
        )
        return np.concatenate(
            [nearest_faces[..., None], receiver_indices[..., None]], axis=-1
        ).reshape(-1, 2)

    def query_edges_from_radius(self, receiver_mesh: "Mesh", radius: float) -> NDArray:
        """Get edges connecting nearest neighboring nodes to receiver nodes.

        Args:
            receiver_mesh: The mesh object to query edges from.
            radius: The radius within which to find neighboring nodes.

        Returns:
            Array of edges connecting nearest neighboring nodes to receiver
                nodes, shaped (N,2)
        """
        nearest_senders = query_radius(self.nodes, receiver_mesh.nodes, radius=radius)
        return form_edges(nearest_senders, np.arange(receiver_mesh.num_nodes))

    #
    def query_edges_from_neighbors(self, receiver_mesh, n_neighbors):
        """Return edges connecting nearest neighboring nodes to receiver nodes.

        Args:
            receiver_mesh: The mesh object to receive edges from.
            n_neighbors: The number of nearest neighbors to consider.

        Returns:
            Array of edges connecting nearest neighboring nodes to receiver
                nodes, shaped (N,2)
        """
        _, nearest_senders = query_nearest(
            self.nodes, receiver_mesh.nodes, n_neighbors=n_neighbors
        )
        return form_edges(nearest_senders, np.arange(receiver_mesh.num_nodes))

    def _invalidate_cache(self):
        self._is_dirty = True
        self._cached_properties = {}

    def reset(self):
        self._nodes_to_keep = np.ones(self._all_nodes.shape[0], dtype=bool)
        self._invalidate_cache()


class RefinableMesh(Mesh):
    """
    An abstract base for meshes that are created by refining a base geometry.

    This class encapsulates the creation pattern:
    1. Get a base set of nodes and faces (`_base`).
    2. Optionally rotate them.
    3. Refine them using a specific strategy (`_refine`).

    Subclasses *must* implement `_base` and `_refine`.
    """

    rotation_axis = "y"
    rotation_angle = 0

    def __init__(
        self,
        nodes=None,
        faces=None,
        refine_factor: int = 1,
        rotate: bool = False,
        refine_by_angle: bool = False,
    ):
        # 1. Get the base geometry from the concrete class (e.g., Icosphere)

        if nodes is None:
            base_nodes, base_faces = self._base()
        else:
            base_nodes, base_faces = nodes, faces

        # 2. Optionally rotate it
        if rotate:
            base_nodes = rotate_nodes(
                base_nodes,
                axis=self.__class__.rotation_axis,
                angle=self.__class__.rotation_angle,
            )

        # 3. Refine the geometry
        nodes, faces = self._refine(
            base_nodes,
            base_faces,
            refine_factor=refine_factor,
            refine_by_angle=refine_by_angle,
        )

        # 4. Initialize the parent Mesh with the final nodes and faces
        super().__init__(nodes, faces)
        self.metadata["factor"] = refine_factor


class TriangularMesh(Mesh):
    """Mixin class for meshes with triangular faces."""

    @staticmethod
    def _refine(nodes, faces, refine_factor, refine_by_angle=False, **kwargs):
        return refine_triangles(
            nodes,
            faces,
            factor=refine_factor,
            angle=refine_by_angle,
            **kwargs,
        )

    def faces2triangles(self, faces: NDArray) -> NDArray:
        return faces


class RectangularMesh(Mesh):
    """Mixin class for meshes with rectangular faces."""

    @staticmethod
    def _refine(nodes, faces, refine_factor, refine_by_angle=False, **kwargs):
        return refine_rectrangles(nodes, faces, **kwargs)

    def faces2triangles(self, faces: NDArray) -> NDArray:
        if faces.ndim != 2 or faces.shape[1] != 4:
            return np.empty((0, 3), dtype=faces.dtype)
        return np.vstack((faces[:, [0, 1, 2]], faces[:, [2, 3, 0]]))


class Icosphere(TriangularMesh, RefinableMesh):
    """
    A triangular mesh generated from a refined icosahedron.
    Rotation angle is chosen to match Graphcast paper.
    """

    rotation_angle = -np.pi / 2 + np.arcsin((1 + np.sqrt(5)) / np.sqrt(12))
    rotation_axis = "y"

    @staticmethod
    def _base() -> Tuple[NDArray, NDArray]:
        """Provides the base 12-node, 20-face icosahedron geometry."""
        phi = (1 + np.sqrt(5)) / 2
        nodes = np.array(
            [
                [0, 1, phi],
                [0, -1, phi],
                [1, phi, 0],
                [-1, phi, 0],
                [phi, 0, 1],
                [-phi, 0, 1],
            ]
        )
        nodes = np.concatenate([nodes, -nodes], axis=0)
        nodes /= np.linalg.norm(nodes, axis=1, keepdims=True)
        faces = np.array(
            [
                [0, 1, 4],
                [0, 2, 3],
                [0, 3, 5],
                [0, 2, 4],
                [0, 1, 5],
                [1, 5, 8],
                [1, 8, 9],
                [2, 4, 11],
                [2, 7, 11],
                [3, 2, 7],
                [3, 7, 10],
                [4, 1, 9],
                [4, 9, 11],
                [5, 3, 10],
                [5, 8, 10],
                [7, 6, 11],
                [8, 6, 10],
                [9, 6, 8],
                [10, 6, 7],
                [11, 6, 9],
            ],
            dtype=int,
        )
        return nodes, faces


class Octasphere(RefinableMesh, TriangularMesh):
    """A triangular mesh generated from a refined octahedron."""

    @staticmethod
    def _base() -> Tuple[NDArray, NDArray]:
        """Provides the base 6-node, 8-face octahedron geometry."""
        vertices = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, -1],
                [0, -1, 0],
                [-1, 0, 0],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 1, 2],
                [5, 2, 3],
                [5, 3, 4],
                [5, 4, 1],
            ],
            dtype=int,
        )
        return vertices, faces


class Cubesphere(RefinableMesh, RectangularMesh):
    """Represents an cubesphere mesh, square-based.

    Attributes:
        rotation_angle: The angle used for rotating the icosphere.
        rotation_axis: The axis around which the icosphere is rotated.
    """

    rotation_angle = np.pi / 4
    rotation_axis = "y"

    @staticmethod
    def _base():
        """
              Create the base cube

              Returns:
                  Tuple (nodes, faces) of shapes (8,3), (6,3)

            (-1,-1,1) 4------------5 (-1,1,1)
                     /|           /|
                    / |          / |
                   /  |         /  |
         (1,-1,1) 0---|--------1 (1,1,1)
           (-1,-1,-1) 7--------|---6 (-1,1,-1)
                  |  /         |  /
                  | /          | /
                  |/           |/
        (1,-1,-1) 3------------2 (1,1,-1)
        """

        nodes = np.array(
            [
                [1, -1, 1],
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
            ]
        )
        nodes = nodes / np.sqrt(3)
        faces = np.array(
            [
                [0, 1, 2, 3],
                [1, 5, 6, 2],
                [5, 4, 7, 6],
                [4, 0, 3, 7],
                [0, 4, 5, 1],
                [3, 2, 6, 7],
            ],
            dtype=int,
        )

        return nodes, faces


class NodesOnlyMesh(Mesh):
    """
    A "mesh" defined only by a set of nodes. Inherits directly from Mesh
    as it does not use the refinement creation pattern.
    """

    def __init__(self, nodes_latlong):
        nodes_xyz = latlong_to_xyz(nodes_latlong)
        faces = np.arange(nodes_xyz.shape[0])[:, np.newaxis].repeat(3, axis=1)
        super().__init__(nodes_xyz, faces)

    def faces2triangles(self, faces: NDArray) -> NDArray:
        return faces


class UniformMesh(NodesOnlyMesh):  # pylint: disable=W0223
    """Mesh of uniformly distributed latitude and longitude"""

    def __init__(self, resolution=1.0):
        self.resolution = resolution
        self.multiplier_long = int(180.0 / resolution)
        self.multiplier_lat = int(90.0 / resolution)
        self.uniform_long = (
            np.arange(-self.multiplier_long, self.multiplier_long, 1) * resolution
        )
        self.uniform_lat = (
            np.arange(-self.multiplier_lat, self.multiplier_lat + 1, 1) * resolution
        )
        self.uniform_coords = (
            np.array(np.meshgrid(self.uniform_lat, self.uniform_long)).reshape(2, -1).T
        )
        super().__init__(self.uniform_coords)

    def reshape(self, values):
        vals = values.T.reshape(
            self.uniform_long.shape[0], self.uniform_lat.shape[0], -1
        ).transpose(1, 0, 2)
        if values.ndim == 1:
            return vals[..., 0]
        return vals


class NestedMeshes:
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

        self._metadata = {
            "factors": factors,
            "cumulative_factors": np.cumprod(factors).tolist(),
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
