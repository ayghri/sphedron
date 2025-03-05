"""
Author: Ayoub Ghriss, dev@ayghri.com
Date: 2024

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import List, Tuple, Type
from numpy.typing import NDArray
import numpy as np
import trimesh


from .utils.refine import refine_triangles
from .utils.refine import refine_rectrangles
from .utils.transform import xyz_to_latlon
from .utils.transform import rotate_nodes
from .utils import faces_to_edges
from .utils import query_nearest
from .utils import connect_nodes


class Mesh:
    """A mesh base class consisting of faces and nodes.
    Each face is a triangle of 3 nodes
    """

    rotation_axis = "y"
    rotation_angle = 0

    def __init__(self, nodes, faces):
        self._metadata = {}
        self._all_nodes = nodes
        self._all_faces = faces
        self._nodes_to_keep: NDArray  # boolean mask for nodes to be kept
        self.reset()

    @classmethod
    def from_base(
        cls,
        refine_factor: int,
        rotate: bool = True,
        refine_by_angle: bool = False,
    ):
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
        base_nodes,
        base_faces,
        refine_factor: int,
        refine_by_angle: bool = False,
    ):
        nodes, triangles = cls.refine(
            base_nodes,
            base_faces,
            refine_factor=refine_factor,
            refine_by_angle=refine_by_angle,
        )
        mesh = cls(nodes, triangles)
        mesh._metadata["factor"] = refine_factor
        return mesh

    @staticmethod
    def refine(
        nodes, faces, refine_factor, refine_by_angle
    ) -> Tuple[NDArray, NDArray]:
        raise NotImplementedError

    @staticmethod
    def _base() -> Tuple[NDArray, NDArray]:
        raise NotImplementedError

    def reset(self):
        """
        Resets the mesh to its initial state by reinitializing the nodes mask
        """
        self._nodes_to_keep = np.ones(self._all_nodes.shape[0], dtype=bool)

    def __repr__(self) -> str:
        return (
            f"Mesh has: #nodes: {self.num_nodes}\n"
            f"          #faces: {self.num_faces}\n"
            f"          #edges: {self.num_edges}\n"
            f"          #edges_unique: {self.edges_unique.shape[0]}\n"
            f"          metadata: {self._metadata}"
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
    def nodes_latlon(self):
        """Returns the nodes of mesh in latitude/longitude format"""
        return xyz_to_latlon(self.nodes)

    @property
    def num_edges(self):
        """Number of edges"""
        return self.edges.shape[0]

    @property
    def num_faces(self):
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
        sender_trimesh = self.build_trimesh()
        receiver_nodes = receiver_mesh.nodes
        _, _, query_triangle_indices = trimesh.proximity.closest_point(
            sender_trimesh, receiver_nodes
        )

        nearest_faces = self.faces[
            self.triangle2face_index(query_triangle_indices)
        ]

        receiver_nodes = np.tile(
            np.arange(receiver_nodes.shape[0])[:, None],
            (1, nearest_faces.shape[1]),
        )
        return np.concatenate(
            [nearest_faces[..., None], receiver_nodes[..., None]], axis=-1
        ).reshape(-1, 2)

    def query_edges_from_radius(
        self, receiver_mesh: "Mesh", radius: float
    ) -> NDArray:
        """Get edges connecting nearest neighboring nodes to receiver nodes.

        Args:
            receiver_mesh: The mesh object to query edges from.
            radius: The radius within which to find neighboring nodes.

        Returns:
            Array of edges connecting nearest neighboring nodes to receiver
                nodes, shaped (N,2)
        """
        nearest_senders = query_nearest(
            self.nodes, receiver_mesh.nodes, radius=radius
        )
        return connect_nodes(
            nearest_senders, list(range(receiver_mesh.num_nodes))
        )

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
        nearest_senders = query_nearest(
            self.nodes, receiver_mesh.nodes, n_neighbors=n_neighbors
        )
        return connect_nodes(
            nearest_senders, list(range(receiver_mesh.num_nodes))
        )


class TriangularMesh(Mesh):
    """Mesh class for which faces are triangles"""

    @staticmethod
    def refine(nodes, faces, refine_factor, refine_by_angle):
        return refine_triangles(
            nodes,
            faces,
            factor=refine_factor,
            use_angle=refine_by_angle,
        )

    def face2triangle_index(self, face_idx):
        return face_idx

    def triangle2face_index(self, triangle_idx):
        return triangle_idx

    @property
    def triangles(self):
        return self.faces


class RectangularMesh(TriangularMesh):
    """Mesh class for which faces are rectangles"""

    @staticmethod
    def refine(nodes, faces, refine_factor, refine_by_angle):
        return refine_rectrangles(
            nodes,
            faces,
            factor=refine_factor,
            use_angle=refine_by_angle,
        )

    def face2triangle_index(self, face_idx, num_faces=None):
        if num_faces is None:
            num_faces = self.num_faces
        return np.stack([face_idx, face_idx + num_faces], axis=-1)

    def triangle2face_index(self, triangle_idx, num_faces=None):
        if num_faces is None:
            num_faces = self.num_faces
        return triangle_idx % num_faces

    def faces2triangles(self, faces):
        return np.r_[faces[:, [0, 1, 2]], faces[:, [2, 3, 0]]]


class NestedMeshes(Mesh):
    """
    A class to create nested meshes, where self.mesh[i+1] is a refined
    self.meshes[i].

    Args:
        factors: A list of integers representing the refinement factors for
            each level of nesting. Each factor be greater than 1, except for
            the first factor
        refine_by_angle: A boolean flag indicating whether to refine the mesh
            based on angle or length. Defaults to False.

    Attributes:
        meshes: A list to hold the created nested meshes.
    """

    _base_mesh_cls: Type[TriangularMesh | RectangularMesh]

    def __init__(self, factors: List[int], refine_by_angle=False):
        assert np.min(factors) >= 1
        self._metadata = {}
        self.meshes = []
        # nodes, faces = self._base_mesh_cls._
        mesh0 = self._base_mesh_cls.from_base(refine_factor=1)
        nodes, faces = mesh0.nodes, mesh0.faces
        for factor in factors:
            self.meshes.append(
                self._base_mesh_cls.from_graph(
                    nodes,
                    faces,
                    refine_factor=factor,
                    refine_by_angle=refine_by_angle,
                )
            )
            nodes, faces = self.meshes[-1].nodes, self.meshes[-1].faces

        super().__init__(None, None)
        self._metadata["depth"] = np.cumprod(factors).tolist()
        self._metadata["factor"] = list(factors)

    def reset(self):
        for mesh in self.meshes:
            mesh.reset()

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

    def mask_nodes(self, nodes_mask: NDArray[np.bool_]):
        if nodes_mask.shape[0] != self.num_nodes:
            raise ValueError(
                f"Nodes mask should have num_nodes={self.num_nodes} entries"
            )
        for mesh in self.meshes:
            mesh.mask_nodes(nodes_mask[: mesh.num_nodes])

    # def face2triangle_index(self, face_idx):
    #     return self.meshes[0].face2triangle_index(
    #         face_idx, num_faces=self.num_faces
    #     )
    #
    # def triangle2face_index(self, triangle_idx):
    #     return self.meshes[0].triangle2face_index(
    #         triangle_idx, num_faces=self.num_faces
    #     )
    #
    # def faces2triangles(self, faces):
    #     return self.meshes[0].faces2triangles(faces)
