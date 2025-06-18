"""
Author: Ayoub Ghriss, dev@ayghri.com

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from .mesh import Icosphere
from .mesh import Octasphere
from .mesh import Cubesphere
from .mesh import NodesOnlyMesh
from .mesh import UniformMesh

from .mesh import NestedIcospheres
from .mesh import NestedOctaspheres
from .mesh import NestedCubespheres

from .transfer import MeshTransfer


__all__ = [
    "Icosphere",
    "Octasphere",
    "Cubesphere",
    "NodesOnlyMesh",
    "UniformMesh",
    "NestedIcospheres",
    "NestedOctaspheres",
    "NestedCubespheres",
    "MeshTransfer",
]
