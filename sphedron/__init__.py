"""
Author: Ayoub Ghriss, dev@ayghri.com

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from .triangular import Icosphere
from .triangular import Octasphere
from .triangular import NestedIcospheres
from .triangular import NestedOctaspheres

from .rectangular import Cubesphere
from .rectangular import NestedCubespheres

from .mesh_transfer import MeshTransfer


__all__ = [
    "Icosphere",
    "Octasphere",
    "NestedIcospheres",
    "NestedOctaspheres",
    "Cubesphere",
    "NestedCubespheres",
    "MeshTransfer",
]
