"""
Author: Ayoub Ghriss, dev@ayghri.com
Date: 2024

License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

from typing import Tuple, Literal
from numpy.typing import NDArray
import numpy as np


def split_edges(
    edge_extremes: NDArray, num_segments: int, use_angle: bool = False
):
    """
    Splits edges defined by pairs of extremities into equally spaced points.

    Given an array of E pairs of extremities, this function returns new points
    that are equally distanced by edge_length/num_segments. If num_segments is
    set to 1, the function returns an empty array. When the extremities are
    located on a sphere, the new points can be split so that they are equally
    distanced by angle when use_angle is set to True, ensuring that the
    normalized points are also equally spaced.

    Args:
        edge_extremes: Extremities of shape (E, 2, 3).
        num_segments: Number of segments to split the edges into.
        use_angle: If set to True, the new points will be equally distanced
            by angle (arc length).

    Returns:
        Array of shape (E * (num_segments - 1), 3) containing the new points.
    """
    if num_segments <= 1:
        return np.array([])
    t = np.arange(1, num_segments) / num_segments
    if use_angle:
        # shape (E,)
        omegas = np.arccos(
            np.sum(edge_extremes[:, 0] * edge_extremes[:, 1], axis=-1)
        )
        sin_om = np.sin(omegas)[:, None]
        u = np.sin(omegas[:, None] * (1 - t[None, :]))
        v = np.sin(omegas[:, None] * t[None, :]) / sin_om
        u = u[:, :, None]  # shape (E,num_segments-1, 1)
        v = v[:, :, None]  # shape (E,num_segments-1, 1)
    else:
        v = t[
            None, :, None
        ]  # interpolation weights, shape (1, num_segments-1, 1)
        u = 1 - v

    nodes_on_edges = u * edge_extremes[:, [0]]  # shape (E,ns,3)
    nodes_on_edges = (
        nodes_on_edges + v * edge_extremes[:, [1]]
    )  # shape(E,ns-1,3)
    nodes_on_edges = nodes_on_edges.reshape(-1, 3)
    return nodes_on_edges


def refine_triangles(
    nodes: NDArray,
    triangles: NDArray,
    factor: int,
    use_angle: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Adapted from https://github.com/vedranaa/icosphere

    Given a base mesh of triangular faces, refine it using a depth factor.

    Args:
        nodes: coordinates representing the mesh nodes, shape (N,3)
        triangles: Indices of each face's nodes, shape (T,3)
        factor: refinement factor that reflects the depth of
            subdivision.
        use_angle: A boolean flag indicating whether to use angle-based
            refinement. See split_edges

    Returns:
        A tuple containing:
            - An array of nodes of the refined mesh, shape (TODO)
            - An array of triangles of the refined mesh. shape (TODO)
    """
    if factor <= 1:
        return nodes, triangles
    # shape [E, 3], where E = F * 3
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]],
        axis=0,
    )

    # sort in alphabetic order and remove duplicates
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    n_triangles = triangles.shape[0]
    n_nodes = nodes.shape[0]
    n_edges = edges.shape[0]
    sub_triangles = np.empty((n_triangles * factor**2, 3), dtype=int)
    new_nodes = np.concatenate(
        [
            nodes,
            np.empty(
                (
                    n_edges * (factor - 1)  # Vertices on edges
                    + n_triangles
                    * (factor - 1)
                    * (factor - 2)
                    // 2,  # nodes on faces, none if factor=2
                    3,
                )
            ),
        ],
        axis=0,
    )

    # Dictionary used to determine the direction of the edge to avoid redundancy
    edge_index = {}
    for i in range(n_edges):
        edge_index[(edges[i, 0], edges[i, 1])] = i
        edge_index[(edges[i, 1], edges[i, 0])] = i

    template = triangle_template(factor)
    ordering = triangles_order(factor)
    reordered_template = ordering[template]

    edge_extremes = nodes[edges]  # shape (E,2,3)
    nodes_on_edges = split_edges(
        edge_extremes,
        num_segments=factor,
        use_angle=use_angle,
    )
    # Step 1: e add (depth-1) nodes per edge
    new_nodes[n_nodes : n_nodes + n_edges * (factor - 1)] = nodes_on_edges

    r = np.arange(factor - 1) + n_nodes
    start_idx = n_edges * (factor - 1) + n_nodes
    num_inside_nodes = (factor - 1) * (factor - 2) // 2
    ref_triangle = np.arange(start_idx, num_inside_nodes + start_idx)
    for triangle_idx in range(n_triangles):
        # First, fixing connectivity. We get hold of the indices of all
        # nodes invoved in this subface: original, on-edges and on-faces.
        triangle_nodes = ref_triangle + triangle_idx * num_inside_nodes
        e_ab = (triangles[triangle_idx, 0], triangles[triangle_idx, 1])
        e_ac = (triangles[triangle_idx, 0], triangles[triangle_idx, 2])
        e_bc = (triangles[triangle_idx, 1], triangles[triangle_idx, 2])
        # -- Already added in Step 1
        # Sorting the nodes on edges in the right order
        # making sure edge is oriented from lower to higher vertex index
        sorted_ab = (edge_index[e_ab] * (factor - 1) + r)[:: e_direction(e_ab)]
        sorted_ac = (edge_index[e_ac] * (factor - 1) + r)[:: e_direction(e_ac)]
        sorted_bc = (edge_index[e_bc] * (factor - 1) + r)[:: e_direction(e_bc)]
        sorted_nodes = np.r_[
            triangles[triangle_idx],
            sorted_ab,
            sorted_ac,
            sorted_bc,
            triangle_nodes,
        ]  # nodes in template order
        # sort nodes in ordering
        sub_triangles[
            triangle_idx * factor**2 : (triangle_idx + 1) * factor**2, :
        ] = sorted_nodes[reordered_template]
        # Now geometry, computing positions of on face nodes.
        new_nodes[triangle_nodes, :] = triangle_interior(
            new_nodes[sorted_ab, :],
            new_nodes[sorted_ac, :],
            use_angle=use_angle,
        )

    new_nodes = new_nodes / np.linalg.norm(new_nodes, axis=1, keepdims=True)

    return (new_nodes, sub_triangles)


def triangle_template(nu: int) -> NDArray[np.int64]:
    """
    Template for linking subfaces                  0
    in a subdivision of a face.                   / \
    Returns faces with vertex                    1---2
    indexing given by reading order.            / \\/ \
                                               3---4---5
                                              / \\/ \\/ \
       Illustration for nu=4:                6---7---8---9
                                            / \\/ \\/ \\/ \
                                           10--11--12--13--14

    Args:
        nu (int): depth for which to generate the faces template

    Returns:
        return faces template of shape $(nu^2 , 3)$
    """

    faces = []
    # looping in layers of triangles
    for i in range(nu):
        vertex0 = i * (i + 1) // 2
        skip = i + 1
        for j in range(i):  # adding pairs of triangles, will not run for i==0
            faces.append(
                [j + vertex0, j + vertex0 + skip, j + vertex0 + skip + 1]
            )
            faces.append([j + vertex0, j + vertex0 + skip + 1, j + vertex0 + 1])
        # adding the last (unpaired, rightmost) triangle
        faces.append([i + vertex0, i + vertex0 + skip, i + vertex0 + skip + 1])

    return np.array(faces)


def triangles_order(nu: int):
    """
    Permutation for ordering of                 0
    face nodes which transformes            / \
    reading-order indexing into indexing      3---6
    first corners nodes, then on-edges    / \\/ \
    nodes, and then on-face nodes     4---12--7
    (as illustrated).                      / \\/ \\/ \
                                          5---13--14--8
                                         / \\/ \\/ \\/ \\
                                        1---9--10--11---2
    Args:
        nu (int): depth for which to generate the ordering

    Returns:
        return ordering of length $(nu+1)(nu+2)/2$
    """

    left = list(range(3, nu + 2))
    right = list(range(nu + 2, 2 * nu + 1))
    bottom = list(range(2 * nu + 1, 3 * nu))
    inside = list(range(3 * nu, (nu + 1) * (nu + 2) // 2))

    o = [0]  # topmost corner
    for i in range(nu - 1):
        o.append(left[i])
        o = o + inside[i * (i - 1) // 2 : i * (i + 1) // 2]
        o.append(right[i])
    o = o + [1] + bottom + [2]

    return np.array(o)


def triangle_interior(ab: NDArray, ac: NDArray, use_angle: bool = False):
    """
    Returns coordinates of the inside nodes (marked by star) for subdivision
    of the face ABC when given coordinates of the on-edge nodesAB[i] and AC[i].                     
             A
            / \
          AB0--AC0
          / \\/ \
        AB1---*--AC1
        / \\/ \\/ \
     AB2---*---*---AC2
      / \\/ \\/ \\/ \
     B-BC1--BC2--BC3-C

    Args:
        AB (array): shape (nu)
        AC (array): [TODO:description]
        use_angle (float): 

    vAB: ndarray, shape(depth-2,3)
    vAC: ndarray, shape(depth-2,3)
    Returns:
        [TODO:return]
    """
    if ab.shape[0] <= 1:
        return None
    nodes = []
    for i in range(1, ab.shape[0]):
        nodes.append(
            split_edges(
                np.concatenate([ab[None, [i]], ac[None, [i]]], axis=1),
                i + 1,
                use_angle,
            )
        )
    all_nodes = np.concatenate(nodes, axis=0)
    return all_nodes


def refine_rectrangles(
    nodes: NDArray,
    rectangles: NDArray,
    factor: int,
    use_length: bool = True,
    use_angle: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Adapted from https://github.com/vedranaa/icosphere
    Given a base mesh, refine it using 1/depth factor

    Args:
        nodes (array): [TODO:description]
        faces (array): [TODO:description]
        depth (array): [TODO:description]

    Returns:
        Vertices and faces of the refined mesh
    """
    if factor <= 1:
        return nodes, rectangles
    # shape [E, 3], where E = F * 3
    edges = np.concatenate(
        [
            rectangles[:, [0, 1]],
            rectangles[:, [1, 2]],
            rectangles[:, [2, 3]],
            rectangles[:, [3, 0]],
        ],
        axis=0,
    )

    # sort in alphabetic order and remove duplicates
    # shape (E,2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    num_squares = rectangles.shape[0]
    num_nodes = nodes.shape[0]
    num_edges = edges.shape[0]
    subsquares = np.empty((num_squares * factor**2, 4), dtype=int)
    new_nodes = np.empty(
        (
            num_nodes
            + num_edges * (factor - 1)  # Vertices on edges
            + num_squares * (factor - 1) ** 2,  # nodes on faces, 1 if factor=2
            3,
        )
    )
    new_nodes[:num_nodes] = nodes

    # Dictionary used to determine the direction of the edge to avoid redundancy
    edge_index = {}
    for i in range(num_edges):
        edge_index[tuple(edges[i])] = i
        edge_index[tuple(edges[i][::-1])] = i

    template = square_template(factor)
    ordering = squares_order(factor)
    reordered_template = ordering[template]

    edge_extremes = nodes[edges]  # shape (E,2,3)
    nodes_on_edges = split_edges(
        edge_extremes,
        num_segments=factor,
        use_angle=use_angle,
    )
    # Step 1: e add (depth-1) nodes per edge
    new_nodes[num_nodes : num_nodes + num_edges * (factor - 1)] = nodes_on_edges
    r = np.arange(factor - 1) + num_nodes
    start_idx = num_edges * (factor - 1) + num_nodes
    num_inside_nodes = (factor - 1) ** 2
    # this will be offset later as indices for the nodes on edges
    ref_square = np.arange(start_idx, num_inside_nodes + start_idx)
    for f in range(num_squares):
        # First, fixing connectivity. We get hold of the indices of all
        # nodes invoved in this subface: original, on-edges and on-faces.
        # T containes the indices of on-faces nodes
        square_nodes = ref_square + f * num_inside_nodes
        e_ab = rectangles[f, [0, 1]]
        e_bc = rectangles[f, [1, 2]]
        e_cd = rectangles[f, [2, 3]]
        e_ad = rectangles[f, [0, 3]]
        # -- Already added in Step 1
        # Sorting the nodes on edges in the right order
        sorted_ab = (edge_index[tuple(e_ab)] * (factor - 1) + r)[
            :: e_direction(e_ab)
        ]
        sorted_bc = (edge_index[tuple(e_bc)] * (factor - 1) + r)[
            :: e_direction(e_bc)
        ]
        sorted_cd = (edge_index[tuple(e_cd)] * (factor - 1) + r)[
            :: e_direction(e_cd)
        ]
        sorted_ad = (edge_index[tuple(e_ad)] * (factor - 1) + r)[
            :: e_direction(e_ad)
        ]
        # --
        sorted_nodes = np.r_[
            rectangles[f],
            sorted_ab,
            sorted_ad,
            sorted_bc,
            sorted_cd,
            square_nodes,
        ]
        subsquares[f * factor**2 : (f + 1) * factor**2, :] = sorted_nodes[
            reordered_template
        ]
        # Now geometry, computing positions of face nodes.
        new_nodes[square_nodes, :] = square_interior(
            new_nodes[sorted_ad, :],
            new_nodes[sorted_bc, :],
            use_length=use_length,
        )
    # normalize nodes to position them on the unit sphere
    new_nodes = new_nodes / np.linalg.norm(new_nodes, axis=1, keepdims=True)

    return (new_nodes, subsquares)


def square_template(nu: int) -> NDArray[np.int64]:
    """
    Template for linking subfaces    0===1===2===3
    in a subdivision of a face.      |   |   |   |
    Returns faces with vertex        4===5===6===7
    indexing given by reading order. |   |   |   |
                                     8===9==10==11
                                     |   |   |   |
       Illustration for nu=3:        12==13==14==15



    Args:
        nu (int): depth for which to generate the faces template

    Returns:
        return faces template of shape $(nu^2 , 3)$
    """

    faces = []
    # looping in layers of squares
    for i in range(nu):
        row0 = i * (nu + 1)  # start 0, 4, 8
        row1 = (i + 1) * (nu + 1)  # start at 4, 8, 12
        for j in range(nu):  # adding (0,1,5,4), (1,2,6,5)....
            faces.append([row0 + j, row0 + j + 1, row1 + j + 1, row1 + j])

    return np.array(faces)


def squares_order(nu: int):
    """
    Permutation for ordering of           0==4===5===1
    face nodes which transformes       |  |   |   |
    reading-order indexing into indexing  6==12==13==8
    first corners nodes, then on-edges |  |   |   |
    nodes, and then on-face nodes   7==14==15==9
    (as illustrated).                     |  |   |   |
                                          3==11==10==2
    Args:
        nu (int): depth for which to generate the ordering

    Returns:
        return ordering of length $(nu+1)**2$
    """

    top = list(range(4, nu + 3))
    left = list(range(nu + 3, 2 * nu + 2))
    right = list(range(2 * nu + 2, 3 * nu + 1))
    bottom = list(range(3 * nu + 1, 4 * nu))[::-1]
    inside = list(range(4 * nu, (nu + 1) ** 2))

    order = [0] + top + [1]  # topmost corner
    for i in range(nu - 1):
        # for j in range(nu):
        order.append(left[i])
        order = order + inside[i * (nu - 1) : (i + 1) * (nu - 1)]
        order.append(right[i])
    order = order + [3] + bottom + [2]
    return np.array(order)


def square_interior(ad: NDArray, bc: NDArray, use_length: bool = True):
    """
     Returns coordinates of the inside nodes (marked by star) for subdivision
     of the face ABCD when given coordinates of the on-edge nodes AD[i]
     and BC[i].
     These should be returned in the correct order as in squares_order

     demo for depth 4
     A==AB0==AB1==AB2==B
     |   |    |    |   |
    AD0==*====*====*==BC0
     |   |    |    |   |
    AD1==*====*====*==BC1
     |   |    |    |   |
    AD2==*====*====*==BC2
     |   |    |    |   |
     D==DC0==DC1==DC2==C

     Args:
         AD: ndarray, shape(depth-1,3)
         BC: ndarray, shape(depth-1,3)
     Returns:
         ndarray of inside nodes, shape ((depth-1)**2, 3)
    """
    assert ad.shape[0] == bc.shape[0]
    extremities = np.concatenate([ad[:, None], bc[:, None]], axis=1)
    return split_edges(
        extremities,
        num_segments=ad.shape[0] + 1,
        use_angle=use_length,
    )


def e_direction(edge) -> Literal[-1, 1]:
    """Determines the direction of an edge based on its node indices.

    Args:
        edge: A tuple containing the indices of the edge nodes.
    Returns:
        -1 if edge[0] > edge[1], 1 otherwise.
    """
    return 1 - 2 * (edge[0] > edge[1])
