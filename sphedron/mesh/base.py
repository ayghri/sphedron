# License: Non-Commercial Use Only
"""
Define the base Mesh class on which the other classes are based
"""

from typing import Tuple
from numpy.typing import NDArray
import numpy as np
import trimesh


from sphedron import refine as _refine
from sphedron import transform as _transform
from sphedron import helpers as _helpers

classes = [
    "Mesh",
    "TriangularMesh",
    "RectangularMesh",
    "NodesOnlyMesh",
]


class Mesh:
    """Flexible base class for spherical meshes.

    Provides core functionality for:
      - Node and face storage
      - Masking of nodes
      - Cached property computation

    Creation pattern:
      - Use from_base() to start from a class-defined base graph.
      - Use from_graph() to start from provided nodes/faces.
      Both optionally rotate and refine before constructing the Mesh.

    Subclasses must implement:
      - base() to supply initial nodes and faces
      - refine() to apply a refinement strategy

    Attributes:
        rotation_axis (str): Axis used for rotations ('x', 'y', or 'z').
        rotation_angle (float): Angle in radians for node rotations.
    """

    rotation_axis = "y"
    rotation_angle = 0

    def __init__(self, nodes: NDArray, faces: NDArray):
        # Enforce explicit construction: require nodes and faces.

        self._all_nodes = np.asarray(nodes)
        self._all_faces = np.asarray(faces, dtype=int)
        self._cached_properties = {}
        self._nodes_to_keep = np.ones(self._all_nodes.shape[0], dtype=bool)
        self.meta = {}

    def reset(self):
        self._nodes_to_keep = np.ones(self._all_nodes.shape[0], dtype=bool)

    @classmethod
    def from_base(
        cls,
        refine_factor: int = 1,
        rotate: bool = True,
        refine_by_angle: bool = False,
    ):
        """Create an instance starting from the class base graph.

        Pipeline:
          base -> optional rotate -> optional refine -> construct

        Args:
            refine_factor: The factor by which to refine the mesh.
            rotate: Whether to rotate base nodes using rotation_axis/angle.
            refine_by_angle: Whether to refine by angle (strategy-specific).

        Returns:
            An instance of the class created from the generated nodes and faces.
        """
        nodes, faces = cls.base()

        if rotate:
            nodes = _transform.rotate_nodes(
                nodes,
                axis=cls.rotation_axis,
                angles=cls.rotation_angle,
            )

        # Always call _refine to keep behavior consistent with previous API.
        nodes, faces = cls.refine(
            nodes,
            faces,
            factor=refine_factor,
            use_angle=refine_by_angle,
        )

        # Use _construct to avoid subclass __init__ side effects.
        instance = cls(nodes=nodes, faces=faces)
        instance.meta["factor"] = refine_factor
        instance.meta["rotate"] = rotate
        instance.meta["angle"] = refine_by_angle
        return instance

    @classmethod
    def from_graph(
        cls,
        nodes: NDArray,
        faces: NDArray,
        refine_factor: int = 1,
        refine_by_angle: bool = False,
        rotate: bool = False,
    ):
        """Create mesh from nodes and faces and optionally rotate/refine.

        Pipeline:
          provided graph -> optional rotate -> optional refine -> construct

        Args:
            nodes (numpy.ndarray[M, ...]): Coordinates of the nodes.
            faces (numpy.ndarray[N, K]): Faces defined by indices into `nodes`.
            refine_factor (int): Factor by which to refine the mesh. Higher
                values yield a more finely subdivided mesh.
            refine_by_angle (bool): If True, refinement considers edge angles.
            rotate (bool): If True, rotate the provided nodes using the class
                rotation parameters before refinement.

        Returns:
            Mesh: A new instance containing the (optionally) refined graph.

        Example:
            >>> from sphedron.mesh import Icosphere
            >>> base_nodes, base_faces = Icosphere._base()
            >>> refined = Icosphere.from_graph(
            ...     base_nodes, base_faces, refine_factor=2
            ... )
            >>> print(refined.num_nodes, refined.num_faces)
        """
        nodes = np.asarray(nodes)
        faces = np.asarray(faces, dtype=int)

        if rotate:
            nodes = _transform.rotate_nodes(
                nodes,
                axis=cls.rotation_axis,
                angles=cls.rotation_angle,
            )

        # Always call _refine to match semantics of from_base.
        nodes, faces = cls.refine(
            nodes,
            faces,
            factor=refine_factor,
            use_angle=refine_by_angle,
        )

        instance = cls(nodes=nodes, faces=faces)
        instance.meta["factor"] = refine_factor
        instance.meta["rotate"] = rotate
        instance.meta["angle"] = refine_by_angle
        return instance

    @staticmethod
    def refine(
        nodes,
        faces,
        factor,
        use_angle=False,
        **kwargs,
    ) -> Tuple[NDArray, NDArray]:
        raise NotImplementedError

    @staticmethod
    def base() -> Tuple[NDArray, NDArray]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"Mesh has: #nodes: {self.num_nodes}\n"
            f"          #faces: {self.num_faces}\n"
            f"          #edges: {self.num_edges}\n"
            f"          #edges_unique: {self.edges_unique.shape[0]}\n"
        )

    @property
    def _node_sorting(self):
        # simple trick that is equivalent to mapping each retained node to its
        # position among all the retained nodes
        return np.cumsum(self._nodes_to_keep) - 1

    @property
    def edges(self):
        """Get the edges of the mesh"""
        edges = _helpers.faces_to_edges(self._all_faces)
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
        """Returns the nodes of mesh in latitude/longitude format (degree)"""
        return _transform.xyz_to_latlong(self.nodes)

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
        nearest_faces = self.faces[
            self.triangle2face_index(query_triangle_indices)
        ]

        receiver_indices = np.tile(
            np.arange(receiver_nodes.shape[0])[:, None],
            (1, nearest_faces.shape[1]),
        )
        return np.concatenate(
            [nearest_faces[..., None], receiver_indices[..., None]], axis=-1
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
        nearest_senders = _helpers.query_radius(
            self.nodes, receiver_mesh.nodes, radius=radius
        )
        return _helpers.form_edges(
            nearest_senders, np.arange(receiver_mesh.num_nodes)
        )

    def query_edges_from_neighbors(self, receiver_mesh, n_neighbors):
        """Return edges connecting nearest neighboring nodes to receiver nodes.

        Args:
            receiver_mesh: The mesh object to receive edges from.
            n_neighbors: The number of nearest neighbors to consider.

        Returns:
            Array of edges connecting nearest neighboring nodes to receiver
                nodes, shaped (N,2)
        """
        _, nearest_senders = _helpers.query_nearest(
            self.nodes, receiver_mesh.nodes, n_neighbors=n_neighbors
        )
        return _helpers.form_edges(
            nearest_senders, np.arange(receiver_mesh.num_nodes)
        )


class NodesOnlyMesh(Mesh):
    """
    A "mesh" defined only by a set of nodes. Inherits directly from Mesh
    as it does not use the refinement creation pattern.
    """

    def __init__(self, nodes_latlong):
        nodes_xyz = _transform.latlong_to_xyz(nodes_latlong)
        faces = np.arange(nodes_xyz.shape[0])[:, np.newaxis].repeat(3, axis=1)
        super().__init__(nodes_xyz, faces)

    def faces2triangles(self, faces: NDArray) -> NDArray:
        return faces


class TriangularMesh(Mesh):
    """Mixin class for meshes with triangular faces."""

    @staticmethod
    def refine(nodes, faces, factor, use_angle=False, **kwargs):
        faces = np.asarray(faces)
        assert faces.shape[1] == 3
        return _refine.refine_triangles(
            nodes,
            faces,
            factor=factor,
            angle=use_angle,
            **kwargs,
        )

    def faces2triangles(self, faces: NDArray) -> NDArray:
        return faces


class RectangularMesh(Mesh):
    """Mixin class for meshes with rectangular faces."""

    @staticmethod
    def refine(nodes, faces, factor, use_angle=False, **kwargs):
        faces = np.asarray(faces)
        assert faces.shape[1] == 4

        return _refine.refine_rectrangles(
            nodes, faces, factor=factor, use_angle=use_angle, **kwargs
        )

    def faces2triangles(self, faces: NDArray) -> NDArray:
        if faces.ndim != 2 or faces.shape[1] != 4:
            return np.empty((0, 3), dtype=faces.dtype)
        return np.vstack((faces[:, [0, 1, 2]], faces[:, [2, 3, 0]]))
