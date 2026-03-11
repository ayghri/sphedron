"""
We define here various meshes based on triangular and rectangular refinement.
"""

from typing import Tuple, Optional
from numpy.typing import NDArray
import numpy as np
import sphedron.transform as _transform

from .base import TriangularMesh
from .base import RectangularMesh
from . import _references as refs


def _normalize_rows(nodes: NDArray) -> NDArray:
    """Return nodes normalized to unit length per row."""
    return nodes / np.linalg.norm(nodes, axis=1, keepdims=True)


class Icosphere(TriangularMesh):
    """
    A triangular mesh generated from a refined icosahedron.
    Rotation angle is chosen to match Graphcast paper.
    """

    rotation_angle = -np.pi / 2 + np.arcsin((1 + np.sqrt(5)) / np.sqrt(12))
    rotation_axis = "y"

    @staticmethod
    def base() -> Tuple[NDArray, NDArray]:
        """Provides the base 12-node, 20-face icosahedron geometry."""
        nodes = _normalize_rows(refs.ICOSAHEDRON_NODES)
        return nodes, refs.ICOSAHEDRON_FACES


class Octasphere(TriangularMesh):
    """A triangular mesh generated from a refined octahedron."""

    @staticmethod
    def base() -> Tuple[NDArray, NDArray]:
        """Provides the base 6-node, 8-face octahedron geometry."""
        return refs.OCTAHEDRON_NODES, refs.OCTAHEDRON_FACES


class Cubesphere(RectangularMesh):
    """Represents an cubesphere mesh, square-based.

    Attributes:
        rotation_angle: The angle used for rotating the icosphere.
        rotation_axis: The axis around which the icosphere is rotated.
    """

    # rotation_angle = np.pi / 4
    # rotation_axis = "y"
    # rotation_angle = np.pi / 4
    # rotation_axis = "y"

    @staticmethod
    def base() -> Tuple[NDArray, NDArray]:
        """Create the base cube geometry.

        Returns:
            Tuple (nodes, faces) of shapes (8, 3) and (6, 4).

        The cube layout::

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

        nodes = refs.CUBE_NODES / np.sqrt(3)
        return nodes, refs.CUBE_FACES


class UniformMesh(RectangularMesh):
    """A rectangular mesh of uniformly distributed latitude and longitude.

    The lat/lon grid is connected with quad faces where each cell connects
    four neighboring grid points.  Longitude wraps around so the last column
    connects back to the first.

    Args:
        resolution: Grid spacing in degrees (default 1.0). Ignored if
            *uniform_lats* and *uniform_longs* are provided.
        uniform_lats: Custom latitude values in degrees. Must be
            provided together with *uniform_longs*.
        uniform_longs: Custom longitude values in degrees. Must be
            provided together with *uniform_lats*.

    Attributes:
        resolution (float): Grid spacing in degrees.
        uniform_lats (NDArray): Latitude values in degrees.
        uniform_longs (NDArray): Longitude values in degrees.
        uniform_latlongs (NDArray): All ``(lat, lon)`` pairs, shape ``(N, 2)``.
    """

    def __init__(
        self,
        resolution=1.0,
        uniform_lats: Optional[NDArray] = None,
        uniform_longs: Optional[NDArray] = None,
    ):
        self.resolution = resolution
        if uniform_lats is None and uniform_longs is None:
            self.uniform_longs = np.arange(resolution / 2, 360, resolution)
            self.uniform_lats = np.arange(-90 + resolution / 2, 90, resolution)
        else:
            if uniform_lats is None or uniform_longs is None:
                raise ValueError(
                    "Provide both uniform_lats and uniform_longs or neither."
                )
            self.uniform_longs = uniform_longs
            self.uniform_lats = uniform_lats

        self.uniform_latlongs = (
            np.array(np.meshgrid(self.uniform_lats, self.uniform_longs))
            .reshape(2, -1)
            .T
        )

        nodes_xyz = _transform.latlong_to_xyz(self.uniform_latlongs)
        faces = self._build_quad_faces(
            len(self.uniform_lats), len(self.uniform_longs)
        )
        super().__init__(nodes_xyz, faces)

    @staticmethod
    def _build_quad_faces(n_lats: int, n_longs: int) -> NDArray:
        """Build quad faces from a lat/lon grid (no longitude wrap).

        Node ordering follows meshgrid convention:
        ``flat_index = lon_idx * n_lats + lat_idx``.

        Args:
            n_lats: Number of latitude points.
            n_longs: Number of longitude points.

        Returns:
            Array of shape ((n_longs - 1) * (n_lats - 1), 4) with quad face
            indices.
        """
        lon_idx, lat_idx = np.meshgrid(
            np.arange(n_longs - 1), np.arange(n_lats - 1), indexing="ij"
        )
        lon_idx = lon_idx.ravel()
        lat_idx = lat_idx.ravel()

        # Four corners of each quad
        bl = lon_idx * n_lats + lat_idx
        tl = lon_idx * n_lats + lat_idx + 1
        tr = (lon_idx + 1) * n_lats + lat_idx + 1
        br = (lon_idx + 1) * n_lats + lat_idx
        return np.column_stack([bl, tl, tr, br])

    def reshape(self, values):
        """Reshape flat node values back to the (lat, lon) grid layout.

        Args:
            values: Flat array of shape (num_nodes,) or (num_nodes, d).

        Returns:
            Array of shape (n_lats, n_lons) or (n_lats, n_lons, d).
        """
        vals = values.T.reshape(
            self.uniform_longs.shape[0], self.uniform_lats.shape[0], -1
        ).transpose(1, 0, 2)
        if values.ndim == 1:
            return vals[..., 0]
        return vals
