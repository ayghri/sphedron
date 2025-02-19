"""
Ayoub Ghriss, ayoub.ghriss@colorado.edu
Non-commercial use.
"""

from typing import List, Tuple, Type
from numpy.typing import NDArray
import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore
import trimesh


from .utils import rotate_nodes
from .utils import xyz_to_latlon
from .utils import latlon_to_xyz
from .utils import faces_to_edges


class Mesh:
    """
    A mesh base class consisting of faces and nodes
    Each face is a triangle of 3 nodes
    """

    rotation_angles = 0.0
    rotation_axes = "x"

    def __init__(self, nodes, faces, rotate: bool = False):
        self.meta = {}
        if rotate:
            nodes = rotate_nodes(
                nodes,
                axis=self.rotation_axes,
                angles=self.rotation_angles,
            )
        self._all_nodes = nodes
        self._all_faces = faces
        self._nodes_to_keep: NDArray  # indices of nodes to be included
        self.reset()

    @staticmethod
    def base() -> Tuple[NDArray, NDArray]:
        raise NotImplementedError

    @staticmethod
    def refine(nodes, faces, factor, **kwargs) -> Tuple[NDArray, NDArray]:
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
        edges = faces_to_edges(self._all_faces)
        # discard edges connected to a masked node
        edges_to_keep = np.all(self._nodes_to_keep[edges], axis=1)
        return self._node_sorting[edges[edges_to_keep]]

    @property
    def edges_symmetric(self) -> NDArray[np.int_]:
        return np.r_[self.edges_unique, self.edges_unique[:, ::-1]]

    @property
    def edges_unique(self):
        return np.unique(np.sort(self.edges, axis=1), axis=0)

    @property
    def faces(self) -> NDArray[np.int_]:
        # only faces whose nodes are all retained
        retained_faces = np.all(self._nodes_to_keep[self._all_faces], axis=1)
        return self._node_sorting[self._all_faces[retained_faces]]

    @property
    def faces_partial(self) -> NDArray:
        retained_faces = np.any(self._nodes_to_keep[self._all_faces], axis=1)
        return self._node_sorting[self._all_faces[retained_faces]]

    @property
    def nodes(self):
        return self._all_nodes[self._nodes_to_keep]

    @property
    def nodes_latlon(self):
        return xyz_to_latlon(self.nodes)

    @property
    def triangles(self):
        raise NotImplementedError

    @property
    def num_edges(self):
        return self.edges.shape[0]

    @property
    def num_faces(self):
        retained_faces = np.all(self._nodes_to_keep[self._all_faces], axis=1)
        return int(np.sum(retained_faces))

    @property
    def num_nodes(self):
        return np.sum(self._nodes_to_keep)

    def triangle_face_index(self, triangle_idx):
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

    def radius_query_edges(
        self,
        target_mesh: "Mesh",
        radius: float,
    ) -> NDArray[np.int_]:
        """
        Return the edges (i,j) where i is the index of the source mesh node s_i
        and j the index of the current mesh node t_j such that s_i is within radius of
        t_j.
        """
        current_indices = cKDTree(target_mesh.nodes).query_ball_point(
            x=self.nodes,
            r=radius,
        )
        current_to_target_edges = []
        for current_v, target_nodes in enumerate(current_indices):
            for target_v in target_nodes:
                current_to_target_edges.append((current_v, target_v))
        return np.array(current_to_target_edges)

    def triangle_query_edges(self, target_mesh: "Mesh") -> NDArray:
        """Returns current(source)-target edge indices where each target node is connected
        to the nodes of the nearest triangle from the current mesh
        Args:
            target_mesh: ...
        Returns:
            np.array of shape (target_mesh.num_nodes*3, 2) containing the edges
        """

        current_trimesh = self.build_trimesh()
        target_nodes = target_mesh.nodes
        _, _, query_face_indices = trimesh.proximity.closest_point(
            current_trimesh, target_nodes
        )

        nearest_triangles = self.triangles[query_face_indices]

        target_nodes = np.tile(np.arange(target_nodes.shape[0])[:, None], (1, 3))
        return np.stack(
            [nearest_triangles[..., None], target_nodes[..., None]], axis=-1
        ).reshape(-1, 2)


class NestedMeshes(Mesh):
    """
    A class to create a stratified icosphere mesh.

    This class generates a mesh composed of multiple icospheres at specified depths,
    allowing for stratification of the geometry. The total mesh is the concatenation
    of all levels.

    Parameters:
    factors (List[int]): A list of integer factors for the icosphere refinement,
        all factors shoud be > 1

    rotate (bool): A flag indicating whether to rotate the icospheres (default is True).
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
        return sum([mesh.num_edges for mesh in self.meshes])

    @property
    def num_faces(self):
        return sum([mesh.num_faces for mesh in self.meshes])

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
        """
        Mask the nodes associated with nodes_mask[i]==True
        Expects a boolean mask of num_nodes size.
        Does not unmask previously masked nodes
        """

        if nodes_mask.shape[0] != self.num_nodes:
            raise ValueError(
                f"Nodes mask should have num_nodes={self.num_nodes} entries"
            )
        for mesh in self.meshes:
            mesh.mask_nodes(nodes_mask[: mesh.num_nodes])


class VerticesOnlyMesh(Mesh):
    def __init__(self, nodes_latlon):
        super().__init__(
            latlon_to_xyz(nodes_latlon),
            np.c_[np.arange(nodes_latlon.shape[0]), np.arange(nodes_latlon.shape[0])],
        )


class UniformMesh(VerticesOnlyMesh):
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

    # def mask_faces(self, faces_mask: NDArray[np.bool_]):
    #     """
    #     Mask faces and their nodes, if the nodes are not associated with other faces
    #     """
    #     if faces_mask.shape[0] != self.num_faces:
    #         raise ValueError(
    #             f"Faces mask should have num_faces={self.num_faces} entries"
    #         )
    #     # self.faces_mask = self.faces_mask[np.logical_not(faces_mask)]
    #     # self.nodes_mask = np.unique(self._all_faces[self.faces_mask])
    #     # nodes_mask =
    #     retained_nodes = np.unique(self.faces[~faces_mask])
    #     nodes_mask = np.ones(self.num_nodes, dtype=bool)
    #     nodes_mask[retained_nodes] = False
    #     self.mask_nodes(nodes_mask)
