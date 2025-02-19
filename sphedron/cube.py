"""
Ayoub Ghriss, dev@ayghri.com
Non-commercial use.
"""

import numpy as np
from .utils import square_refine
from .icosphere import Icosphere
from .icosphere import NestedMeshes


class Cubesphere(Icosphere):
    rotation_angle = np.pi / 4
    rotation_axis = "y"

    @staticmethod
    def refine(nodes, faces, factor, use_angle=False, **kwargs):
        return square_refine(
            nodes,
            squares=faces,
            factor=factor,
            use_length=use_angle,
        )

    @staticmethod
    def base():
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

    @property
    def triangles(self):
        faces = self.faces
        triangles = np.r_[faces[:, [0, 1, 2]], faces[:, [2, 3, 0]]]
        return triangles

    def triangle_face_index(self, triangle_idx):
        return triangle_idx % self.num_faces


class NestedCubespheres(NestedMeshes):
    base_mesh_cls = Cubesphere

    def triangle_face_index(self, triangle_idx):
        return triangle_idx % self.num_faces
