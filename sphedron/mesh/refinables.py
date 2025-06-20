"""
License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import Tuple
from numpy.typing import NDArray
import numpy as np
import sphedron.transform as _transform

from .base import TriangularMesh
from .base import RectangularMesh
from .base import Mesh
from .base import RefinableMesh


class Icosphere(TriangularMesh, RefinableMesh):
    """
    A triangular mesh generated from a refined icosahedron.
    Rotation angle is chosen to match Graphcast paper.
    """

    rotation_angle = -np.pi / 2 + np.arcsin((1 + np.sqrt(5)) / np.sqrt(12))
    rotation_axis = "y"

    @staticmethod
    def _base() -> Tuple[NDArray, NDArray]:
        """Provides the base 12-node, 20-face icosahedron geometry."""
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
        nodes /= np.linalg.norm(nodes, axis=1, keepdims=True)
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


class Octasphere(RefinableMesh, TriangularMesh):
    """A triangular mesh generated from a refined octahedron."""

    @staticmethod
    def _base() -> Tuple[NDArray, NDArray]:
        """Provides the base 6-node, 8-face octahedron geometry."""
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


class Cubesphere(RefinableMesh, RectangularMesh):
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



class UniformMesh(Mesh):  # pylint: disable=W0223
    """Mesh of uniformly distributed latitude and longitude"""

    def __init__(self, resolution=1.0):
        self.resolution = resolution
        self.multiplier_long = int(180.0 / resolution)
        self.multiplier_lat = int(90.0 / resolution)
        self.uniform_long = (
            np.arange(-self.multiplier_long, self.multiplier_long, 1)
            * resolution
        )
        self.uniform_lat = (
            np.arange(-self.multiplier_lat, self.multiplier_lat + 1, 1)
            * resolution
        )
        self.uniform_latlons = (
            np.array(np.meshgrid(self.uniform_lat, self.uniform_long))
            .reshape(2, -1)
            .T
        )

        nodes_xyz = _transform.latlong_to_xyz(self.uniform_latlons)
        faces = np.arange(nodes_xyz.shape[0])[:, np.newaxis].repeat(3, axis=1)
        super().__init__(nodes_xyz, faces)

    def reshape(self, values):
        vals = values.T.reshape(
            self.uniform_long.shape[0], self.uniform_lat.shape[0], -1
        ).transpose(1, 0, 2)
        if values.ndim == 1:
            return vals[..., 0]
        return vals
