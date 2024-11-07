"""
Ayoub Ghriss, dev@ayghri.com
Non-commercial use.
"""

import numpy as np
from .icosphere import Icosphere


class Octasphere(Icosphere):
    @staticmethod
    def base():
        """
        Create the base octahedron

        Returns:
            Tuple (vertices, faces) of shapes (6,3), (8,3)
        """

        vertices = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, -1],
                [0, -1, 0],
                [-1, 0, 0],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 1, 2],
                [5, 2, 3],
                [5, 3, 4],
                [5, 4, 1],
            ],
            dtype=int,
        )

        return vertices, faces
