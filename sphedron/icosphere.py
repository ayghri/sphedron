"""
Ayoub Ghriss, dev@ayghri.com
Non-commercial use.
"""

from typing import List
import numpy as np
import time

from .mesh import Mesh
from .utils.mesh import triangle_refine


class Icosphere(Mesh):
    """

    Attributes
    ----------
    rotation_angle :
    rotation_axis :

    """

    rotation_angle = (np.pi - 2 * np.arcsin((1 + np.sqrt(5)) / 2 / np.sqrt(3))) / 2
    rotation_axis = "y"

    def __init__(
        self,
        depth: int,
        rotate=True,
        use_angle=False,
        normalize=True,
    ):
        start = time.time()
        base_vertices, base_faces = self.base()
        vertices, faces = triangle_refine(
            base_vertices,
            base_faces,
            factor=depth,
            use_angle=use_angle,
            normalize=normalize,
        )
        super().__init__(vertices=vertices, faces=faces, rotate=rotate)
        self.meta["compute time (ms)"] = 1000 * (time.time() - start)
        self.meta["depth"] = depth

    @staticmethod
    def base():
        """
        Create the base icosphere
        Returns:
            Tuple (vertices, faces) for shape (12,3), (20,3)
        """

        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array(
            [
                [0, 1, phi],
                [0, -1, phi],
                [1, phi, 0],
                [-1, phi, 0],
                [phi, 0, 1],
                [-phi, 0, 1],
            ]
        )
        vertices = np.concatenate([vertices, -vertices], axis=0)
        vertices = vertices / np.sqrt(1 + phi**2)
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
        return vertices, faces


class StratifiedIcospheres(Mesh):
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
        # we build the icospheres in a stratified way
        # where next refinement takes the current one
        # the refinement keeps the current vertices and adds new ones
        # so the vertices of the last refinement contains the vertices of all
        assert np.min(factors) >= 1
        all_faces = []
        vertices, faces = Icosphere.base()
        for factor in factors:
            vertices, faces = triangle_refine(vertices, faces, factor=factor)
            all_faces.append(faces)
        faces = np.concatenate(all_faces, axis=0)
        super().__init__(vertices, faces, rotate=rotate)
        self.meta["depths"] = np.cumprod(factors)
