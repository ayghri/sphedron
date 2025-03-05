"""
Author: Ayoub Ghriss, dev@ayghri.com
Date: 2024

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

    # def __init__(
    #     self,
    #     refine_factor: int = 1,
    #     refine_by_angle=False,
    #     rotate=True,
    #     nodes=None,
    #     faces=None,
    # ):
    #
    #     start = time.time()
    #     if nodes is None or faces is None:
    #         nodes, faces = self._base()
    #
    #     if rotate:
    #         nodes = rotate_nodes(
    #             nodes,
    #             axis=self.rotation_axis,
    #             angle=self.rotation_angle,
    #         )
    #
    #     super().__init__(
    #         nodes,
    #         faces,
    #         refine_factor=refine_factor,
    #         refine_by_angle=refine_by_angle,
    #     )
    #
    #     self._metadata["compute time (ms)"] = 1000 * (time.time() - start)
    #     self._metadata["factor"] = refine_factor

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
