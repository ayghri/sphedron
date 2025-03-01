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


from .base import Mesh
from .utils.transform import latlon_to_xyz


class NodesOnlyMesh(Mesh):  # pylint: disable=W0223
    """Mesh where only nodes are provided. creates fakes self-triangles"""

    def __init__(self, nodes_latlon):
        super().__init__(
            latlon_to_xyz(nodes_latlon),
            np.c_[
                np.arange(nodes_latlon.shape[0]),
                np.arange(nodes_latlon.shape[0]),
            ],
        )

    @property
    def triangles(self):
        return self.faces


class UniformMesh(NodesOnlyMesh):  # pylint: disable=W0223
    """Mesh of uniformly distributed latitude and longitude"""

    def __init__(self, resolution=1.0):
        multiplier_long = int(180.0 / resolution)
        multiplier_lat = int(90.0 / resolution)
        uniform_long = (
            np.arange(-multiplier_long, multiplier_long, 1) * resolution
        )
        uniform_lat = (
            np.arange(-multiplier_lat, multiplier_lat + 1, 1) * resolution
        )
        uniform_coords = (
            np.array(np.meshgrid(uniform_long, uniform_lat)).reshape(2, -1).T
        )
        super().__init__(uniform_coords)
