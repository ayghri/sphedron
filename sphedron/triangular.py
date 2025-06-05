"""
Author: Ayoub Ghriss, dev@ayghri.com

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

import numpy as np

from .base import TriangularMesh
from .base import NestedMeshes


class Icosphere(TriangularMesh):
    """Represents an icosphere mesh, triangle-based.

    The rotation attributes are chosen to match GraphCast implementation.
    See `GraphCast: Learning skillful medium-range global weather forecasting`:
    https://arxiv.org/abs/2212.12794

    Attributes:
        rotation_angle: The angle used for rotating the icosphere.
        rotation_axis: The axis around which the icosphere is rotated.
    """

    rotation_angle = -np.pi / 2 + np.arcsin((1 + np.sqrt(5)) / np.sqrt(12))
    rotation_axis = "y"

    @staticmethod
    def _base():
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


class NestedIcospheres(NestedMeshes, TriangularMesh):
    """Nested icospheres, where self.mesh[i+1] is a refined self.meshes[i]."""

    _base_mesh_cls = Icosphere


class Octasphere(Icosphere):
    """Mesh based on the octahedron"""

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


class NestedOctaspheres(NestedMeshes, TriangularMesh):
    """Nested Octaspheres meshes"""

    _base_mesh_cls = Octasphere
