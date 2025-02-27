"""
Ayoub Ghriss, ayoub.ghriss@colorado.edu
Non-commercial use.
"""

import numpy as np
import time

from .mesh import Mesh
from .mesh import NestedMeshes
from .utils import triangle_refine


class Icosphere(Mesh):
    """

    Attributes
    ----------
    rotation_angle :
    rotation_axis :


    """

    rotation_angles = np.pi / 2 - np.arcsin((1 + np.sqrt(5)) / np.sqrt(12))
    rotation_axes = "y"

    def __init__(
        self,
        factor: int = 1,
        rotate=True,
        use_angle=False,
        nodes=None,
        faces=None,
        **kwargs,
    ):

        self.use_angle = use_angle

        start = time.time()
        if nodes is None or faces is None:
            base_nodes, base_faces = self.base()
        else:
            base_nodes = nodes
            base_faces = faces
        if factor > 1:
            nodes, faces = self.refine(
                base_nodes, base_faces, factor=factor, use_angle=use_angle
            )

        super().__init__(nodes, faces, rotate=rotate, **kwargs)

        self.meta["compute time (ms)"] = 1000 * (time.time() - start)
        self.meta["factor"] = factor

    @staticmethod
    def refine(nodes, faces, factor, use_angle=False, **kwargs):
        return triangle_refine(nodes, faces, factor=factor, use_angle=use_angle)

    @staticmethod
    def base():
        """
        Create the base icosphere
        Returns:
            Tuple (nodes, faces) for shape (12,3), (20,3)
        """

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
        nodes = nodes / np.linalg.norm(nodes, axis=1, keepdims=True)
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

    @property
    def triangles(self):
        return self.faces

    def triangle2face_index(self, triangle_idx):
        return triangle_idx

    def face2triangle_index(self, face_idx):
        return face_idx


class NestedIcospheres(NestedMeshes):
    base_mesh_cls = Icosphere

    def triangle_face_index(self, triangle_idx):
        return triangle_idx
