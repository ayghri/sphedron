"""
License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from .base import NodesOnlyMesh
from .refinables import Cubesphere
from .refinables import Icosphere
from .refinables import Octasphere
from .refinables import UniformMesh

from .nested import NestedCubespheres
from .nested import NestedIcospheres
from .nested import NestedOctaspheres

__all__ = [
    "NodesOnlyMesh",
    "Cubesphere",
    "Icosphere",
    "Octasphere",
    "NestedCubespheres",
    "NestedIcospheres",
    "NestedOctaspheres",
    "UniformMesh",
]
