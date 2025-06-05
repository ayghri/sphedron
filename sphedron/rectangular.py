"""
Author: Ayoub Ghriss, dev@ayghri.com

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

import numpy as np

from .base import RectangularMesh
from .base import NestedMeshes


class Cubesphere(RectangularMesh):
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


class NestedCubespheres(NestedMeshes, Cubesphere):
    """Nested cubespheres, where self.mesh[i+1] is a refined self.meshes[i]."""

    _base_mesh_cls = Cubesphere
