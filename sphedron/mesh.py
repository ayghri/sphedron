"""
Ayoub Ghriss, ayoub.ghriss@colorado.edu
Non-commercial use.
"""

from typing import List, Tuple, Type
from numpy.typing import NDArray
import numpy as np
import trimesh


from .utils import rotate_nodes
from .utils import xyz_to_latlon
from .utils import latlon_to_xyz
from .utils import faces_to_edges
from .utils import query_nearest


class Mesh:
    """
    A mesh base class consisting of faces and nodes
    Each face is a triangle of 3 nodes
    """

    rotation_angle = 0.0
    rotation_axes = "x"

    def __init__(self, nodes, faces, rotate: bool = False):
        self.meta = {}
        if rotate:
            nodes = rotate_nodes(
                nodes,
                axis=self.rotation_axes,
                angle=self.rotation_angle,
            )
        self._all_nodes = nodes
        self._all_faces = faces
        self._nodes_to_keep: NDArray  # boolean mask for nodes to be kept
        self.reset()

    @staticmethod
    def base() -> Tuple[NDArray, NDArray]:
        """Create base mesh that will be refined later"""
        raise NotImplementedError

    @staticmethod
    def refine(nodes, faces, factor, **kwargs) -> Tuple[NDArray, NDArray]:
        """Takes the nodes and faces of the mesh and refines them by factor"""
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
            f"          metadata: {self.meta}"
        )

    @property
    def _node_sorting(self):
        # simple trick that is equivalent to mapping each retained node to its
        # position among all the retained nodes
        return np.cumsum(self._nodes_to_keep) - 1

    @property
    def edges(self):
        """Returns the edges of the mesh"""
        edges = faces_to_edges(self._all_faces)
        # discard edges connected to a masked node
        edges_to_keep = np.all(self._nodes_to_keep[edges], axis=1)
        return self._node_sorting[edges[edges_to_keep]]

    @property
    def edges_symmetric(self) -> NDArray[np.int_]:
        """Returns the edges of the undirected mesh"""
        return np.r_[self.edges_unique, self.edges_unique[:, ::-1]]

    @property
    def edges_unique(self):
        """Returns the directed edges of the graph, from lower index to higher index nodes"""
        return np.unique(np.sort(self.edges, axis=1), axis=0)

    @property
    def faces(self) -> NDArray[np.int_]:
        """Returns the faces of the mesh, can be rectangles or triangles or more"""
        # only faces whose nodes are all retained
        retained_faces = np.all(self._nodes_to_keep[self._all_faces], axis=1)
        return self._node_sorting[self._all_faces[retained_faces]]

    @property
    def faces_partial(self) -> NDArray:
        """Returns the faces of the mesh including those with partially masked nodes"""
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
        """Number of triangles"""
        raise NotImplementedError

    def triangle2face_index(self, triangle_idx):
        """Convert triangle index to face index, identity for triangle based meshes"""
        raise NotImplementedError

    def face2triangle_index(self, face_idx):
        """Convert face index to triangle index, identity for triangle based meshes"""
        raise NotImplementedError

    def mask_nodes(self, nodes_mask: NDArray[np.bool_]):
        """
        Mask the nodes associated with nodes_mask[i]==True
        Expects a boolean mask of num_nodes size.
        Does not unmask previously masked nodes
        """

        if nodes_mask.shape[0] != self.num_nodes:
            raise ValueError(
                f"Nodes mask should have num_nodes={self.num_nodes} entries"
            )
        assert self.num_nodes == nodes_mask.shape[0]
        self._nodes_to_keep[self._nodes_to_keep] = ~nodes_mask

    def build_trimesh(self):
        """
        return Trimesh instance from the current nodes and faces

        """

        mesh = trimesh.Trimesh(self.nodes, self.triangles)
        mesh.fix_normals()
        return mesh

    def query_edges_from_faces(self, receiver_mesh: "Mesh") -> NDArray:
        """Return edges connecting nodes of the nearest face to receiver nodes"""
        sender_trimesh = self.build_trimesh()
        receiver_nodes = receiver_mesh.nodes
        _, _, query_triangle_indices = trimesh.proximity.closest_point(
            sender_trimesh, receiver_nodes
        )

        nearest_faces = self.faces[self.triangle2face_index(query_triangle_indices)]

        receiver_nodes = np.tile(
            np.arange(receiver_nodes.shape[0])[:, None], (1, nearest_faces.shape[1])
        )
        return np.concatenate(
            [nearest_faces[..., None], receiver_nodes[..., None]], axis=-1
        ).reshape(-1, 2)

    def query_edges_from_neighbors(
        self,
        receiver_mesh: "Mesh",
        radius: float = -1.0,
        n_neighbors: int = -1,
    ):
        """Return edges connecting nearest neighboring nodes to receiver nodes"""
        nearest_senders = query_nearest(
            self.nodes, receiver_mesh.nodes, radius=radius, n_neighbors=n_neighbors
        )[0]
        s2r_edges = []
        for s_is, r_i in zip(nearest_senders, range(receiver_mesh.num_nodes)):
            for s_i in s_is:
                s2r_edges.append([s_i, r_i])
        return np.array(s2r_edges)


class NestedMeshes(Mesh):  # pylint: disable=W0223
    """
    A class to create nested meshes, where self.mesh[i+1] is a refined self.meshes[i]
    """

    base_mesh_cls: Type[Mesh]

    def __init__(self, factors: List[int], rotate=True, **kwargs):
        assert np.min(factors) >= 1
        self.meta = {}
        self.factors = factors
        self.meshes = []
        nodes, faces = self.base_mesh_cls.base()
        for factor in factors:
            nodes, faces = self.base_mesh_cls.refine(
                nodes, faces, factor=factor, **kwargs
            )
            self.meshes.append(
                self.base_mesh_cls(nodes=nodes, faces=faces, rotate=rotate)
            )
        super().__init__(None, None)
        self.meta["depth"] = np.cumprod(factors).tolist()
        self.meta["factor"] = list(factors)

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
    def triangles(self):
        return np.concatenate([mesh.triangles for mesh in self.meshes], axis=0)

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


class NodesOnlyMesh(Mesh):  # pylint: disable=W0223
    """Mesh where only nodes are provided. creates fakes self-triangles"""

    def __init__(self, nodes_latlon):
        super().__init__(
            latlon_to_xyz(nodes_latlon),
            np.c_[np.arange(nodes_latlon.shape[0]), np.arange(nodes_latlon.shape[0])],
        )

    @property
    def triangles(self):
        return self.faces


class UniformMesh(NodesOnlyMesh):  # pylint: disable=W0223
    """Mesh of uniformly distributed latitude and longitude"""

    def __init__(self, resolution=1.0):
        multiplier_long = int(180.0 / resolution)
        multiplier_lat = int(90.0 / resolution)
        uniform_long = np.arange(-multiplier_long, multiplier_long, 1) * resolution
        uniform_lat = np.arange(-multiplier_lat, multiplier_lat + 1, 1) * resolution
        uniform_coords = (
            np.array(np.meshgrid(uniform_long, uniform_lat)).reshape(2, -1).T
        )
        super().__init__(uniform_coords)
        self.uniform_long = uniform_long
        self.uniform_lat = uniform_lat
