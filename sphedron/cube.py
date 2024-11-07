"""
Ayoub Ghriss, dev@ayghri.com
Non-commercial use.
"""

from typing import List
import numpy as np
from .mesh import Mesh
from .utils.mesh import rotate_vertices, square_refine


class Cubesphere(Mesh):
    rotation_angle = np.pi / 4
    rotation_axis = "y"

    def __init__(self, depth: int, rotate=False, use_angle=False, normalize=True):
        base_vertices, base_squares = self.base()
        vertices, squares = square_refine(
            base_vertices,
            base_squares,
            factor=depth,
            use_length=use_angle,
            normalize=normalize,
        )
        # self.squares = squares
        if rotate:
            vertices = rotate_vertices(
                vertices, axis=self.rotation_axis, angle=self.rotation_angle
            )
        super().__init__(vertices=vertices, faces=squares)
        self.meta["depth"] = depth

    @staticmethod
    def base():
        """
              Create the base cube

              Returns:
                  Tuple (vertices, faces) of shapes (8,3), (6,3)

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

        vertices = np.array(
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
        vertices = vertices / np.sqrt(3)
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

        return vertices, faces

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def num_faces(self):
        return self.faces.shape[0]

    @property
    def triangles(self):
        faces = self.faces
        triangles = np.r_[faces[:, [0, 1, 2]], faces[:, [2, 3, 0]]]
        return triangles


class StratifiedCubeSphere(Mesh):
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

    def __init__(self, factors: List[int], rotate=True):
        assert np.min(factors) >= 1
        all_faces = []
        vertices, faces = Cubesphere.base()
        for factor in factors:
            vertices, faces = square_refine(
                vertices, faces, factor=factor, normalize=False
            )
            all_faces.append(faces)
        faces = np.concatenate(all_faces, axis=0)
        triangles = np.r_[faces[:, :3], np.c_[faces[:, 2:], faces[:, 0]]]
        super().__init__(vertices=vertices, faces=triangles, rotate=rotate)
        self.meta["depth"] = np.cumprod(factors)
