"""
Author: Ayoub Ghriss, dev@ayghri.com

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from .mesh.refinables import Cubesphere
from .mesh.refinables import Icosphere
from .mesh.refinables import Octasphere
from .mesh.refinables import UniformMesh

from .mesh.nested import NestedCubespheres
from .mesh.nested import NestedIcospheres
from .mesh.nested import NestedOctaspheres

from .transfer import MeshTransfer

__all__ = [
    "Cubesphere",
    "Icosphere",
    "Octasphere",
    "UniformMesh",
    "NestedCubespheres",
    "NestedIcospheres",
    "NestedOctaspheres",
    "MeshTransfer",
]
