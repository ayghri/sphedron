# License: Non-Commercial Use Only
#
# Permission is granted to use, copy, modify, and distribute this software
# for non-commercial purposes only, with attribution to the original author.
# Commercial use requires explicit permission.
#
# This software is provided "as is", without warranty of any kind.
"""Reference geometry constants for base polyhedra."""

import numpy as np

phi_icosphere = (1 + np.sqrt(5)) / 2

GRAPHCAST_ROTATION = {
    "rotation_axes": "y",
    "rotation_angles": -np.pi / 2 + np.arcsin((1 + np.sqrt(5)) / np.sqrt(12)),
}


ICOSAHEDRON_NODES = np.concatenate(
    [
        np.array(
            [
                [0, 1, phi_icosphere],
                [0, -1, phi_icosphere],
                [1, phi_icosphere, 0],
                [-1, phi_icosphere, 0],
                [phi_icosphere, 0, 1],
                [-phi_icosphere, 0, 1],
            ]
        ),
        -np.array(
            [
                [0, 1, phi_icosphere],
                [0, -1, phi_icosphere],
                [1, phi_icosphere, 0],
                [-1, phi_icosphere, 0],
                [phi_icosphere, 0, 1],
                [-phi_icosphere, 0, 1],
            ]
        ),
    ],
    axis=0,
)
ICOSAHEDRON_FACES = np.array(
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

OCTAHEDRON_NODES = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, -1],
        [0, -1, 0],
        [-1, 0, 0],
    ]
)
OCTAHEDRON_FACES = np.array(
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

CUBE_NODES = np.array(
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
CUBE_FACES = np.array(
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
