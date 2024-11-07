from numpy import typing as npt
import numpy as np
from scipy import spatial
import trimesh


from .utils.mesh import rotate_vertices
from .utils.mesh import cartesian_to_latlon
from .utils.mesh import latlon_to_cartesian


class Mesh:
    """
    A mesh base class consisting of faces and vertices
    Each face is a triangle of 3 vertices
    """

    rotation_angle = 0.0
    rotation_axis = "x"

    def __init__(
        self,
        vertices: npt.NDArray,
        faces: npt.NDArray[np.int_],
        cartesian_vertices=True,
        rotate: bool = False,
    ):
        self.meta = {}
        if rotate:
            vertices = rotate_vertices(
                vertices,
                axis=self.rotation_axis,
                angle=self.rotation_angle,
            )
        self._all_vertices = vertices
        self._all_faces = faces
        self.vertices_subset: npt.NDArray  # indices of vertices to be included
        self.vertices_sorting: npt.NDArray  # sorting of the vertices
        self.faces_subset: npt.NDArray  # indices of the faces to include in the mesh
        if not cartesian_vertices:
            self._all_vertices = latlon_to_cartesian(vertices)
        self.reset()

    def reset(self):
        """
        Resets the mesh to its initial state by reinitializing the vertices and
        faces subsets.
        This method updates the vertices and faces subsets to include all available
        vertices and faces, and reconstructs the mesh using the Trimesh class.
        """
        self.vertices_subset = np.arange(self._all_vertices.shape[0])
        self.vertices_sorting = np.arange(self._all_vertices.shape[0])
        self.faces_subset = np.arange(self._all_faces.shape[0])

    def __repr__(self) -> str:
        return f"Mesh has:\
                \n\t #vertices: {self.num_vertices}\
                \n\t #faces: {self.num_faces},\
                \n\t #edges: {self.edges.shape[0]},\
                \n\t #edges_unique: {self.edges_unique.shape[0]},\
                \n\tmeta: {self.meta}"

    @property
    def vertices(self):
        return self._all_vertices[self.vertices_subset]

    @property
    def num_vertices(self):
        return self.vertices_subset.shape[0]

    @property
    def triangles(self):
        return self.faces

    @property
    def faces(self) -> npt.NDArray[np.int_]:
        return self.vertices_sorting[self._all_faces[self.faces_subset]]

    @property
    def num_faces(self):
        return self.faces_subset.shape[0]

    @property
    def vertices_latlon(self):
        return cartesian_to_latlon(self.vertices)

    @property
    def edges(self):
        """
        Get the edges of the mesh, with possibly redundant ones

        Returns:
            numpy.ndarray:  shape (num_edges, 2), where the connected vertices are
            (edges[i,0], edges[i,1])
        """
        num_edges_per_face = self._all_faces.shape[1]
        return np.concatenate(
            [
                self.faces[:, [i, (i + 1) % num_edges_per_face]]
                for i in range(num_edges_per_face)
            ],
            axis=0,
        )

    @property
    def edges_unique(self):
        """
        Get the unique edges of the mesh

        Returns:
            numpy.ndarray:  shape (num_edges_unique, 2), where the connected vertices
            are (edges[i,0], edges[i,1])
        """
        return np.unique(np.sort(self.edges, axis=1), axis=0)

    @property
    def edges_symmetric(self) -> npt.NDArray[np.int_]:
        """
        Get the symmetric edges of the mesh, so that it includes [v_i,v_j] and
        [v_j,v_i], without redundancy.

        Returns:
            numpy.ndarray:  shape (2*num_edges_unique, 2), where the connected vertices are
            (edges[i,0], edges[i,1])
        """
        return np.r_[self.edges_unique, self.edges_unique[:, ::-1]]

    def exclude_faces(self, faces_mask: npt.NDArray[np.bool_]):
        """
        Mask faces and their associated vertices.
        """
        self.faces_subset = self.faces_subset[np.logical_not(faces_mask)]
        self.vertices_subset = np.unique(self._all_faces[self.faces_subset])
        self.vertices_sorting = -np.ones(self._all_vertices.shape[0], dtype=np.int_)
        self.vertices_sorting[self.vertices_subset] = np.arange(
            self.vertices_subset.shape[0]
        )

        self.trimesh = trimesh.Trimesh(
            self._all_vertices[self.vertices_subset],
            self.vertices_sorting[self._all_faces[self.faces_subset]],
        )

    def exclude_faces_by_vertex(self, vertices_mask: npt.NDArray[np.bool_]):
        """
        Mask the faces associated with the provided masked vertices
        Expects a boolean mask.
        """
        faces = self.faces
        assert self.num_vertices == vertices_mask.shape[0]
        faces_to_exclude = np.any(vertices_mask[faces], axis=1)
        self.exclude_faces(faces_to_exclude)

    def exclude_vertices(self, vertices_mask: npt.NDArray[np.bool_]):
        """
        Mask the faces associated with the provided masked vertices
        Expects a boolean mask.
        """
        vertices_subset = self.vertices_subset[np.logical_not(vertices_mask)]
        # mask faces when its entire vertices are masked:
        vertices_sorting = -np.ones(self._all_vertices.shape[0], dtype=np.int_)
        vertices_sorting[vertices_subset] = np.arange(vertices_subset.shape[0])
        # exclude faces for which at least 2 vertices have been masked
        # this is not necessary, but might speed up things if the mesh is too fine
        faces_to_exclude = (
            np.sum(vertices_sorting[self._all_faces[self.faces_subset]] < 0, axis=1)
            >= 2
        )
        self.exclude_faces(faces_to_exclude)
        self.vertices_subset = vertices_subset
        self.vertices_sorting = vertices_sorting

    def from_source_edges(
        self,
        source_mesh: "Mesh",
        radius: float,
    ) -> npt.NDArray[np.int_]:
        indices = spatial.cKDTree(self.vertices).query_ball_point(
            x=source_mesh.vertices,
            r=radius,
        )
        source_to_mesh_edges = []
        for source_v, mesh_v in enumerate(indices):
            for v in mesh_v:
                source_to_mesh_edges.append((source_v, v))
        return np.array(source_to_mesh_edges).T

    def build_trimesh(self):
        mesh = trimesh.Trimesh(self.vertices, self.triangles)
        mesh.fix_normals()
        return mesh


class VerticesOnlyMesh(Mesh):
    def __init__(self, vertices_latlon):
        super().__init__(
            vertices_latlon,
            np.zeros_like((vertices_latlon, 3)),
            cartesian_vertices=False,
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
