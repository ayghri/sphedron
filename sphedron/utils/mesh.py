from typing import Literal, Tuple
import numpy as np
import numpy.typing as npt
from scipy.spatial import transform
from sklearn.neighbors import NearestNeighbors


def rotate_vertices(vertices: npt.NDArray, axis: str, angle: float):
    """
    Rotate the mesh
    Args:
        vertices: (np.array) shape (num_vertices, 3)
    Returns:
        vertices: (np.array) rotated vertices of shape (num_vertices, 3)

    """
    # return transform.Rotation.from_euler(seq=axis, angles=rotation_angle).apply(
    #     vertices
    # )
    rotation = transform.Rotation.from_euler(seq=axis, angles=angle)
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)
    return vertices


def split_edges(edge_extremes: npt.NDArray, num_segments: int, use_angle: bool):
    """
    Given an array of E pairs of extremities, returns the new points that
    are equally distanced by edge_length/num_segments. (num_segments=1 return nothing)
    When the extremities are located on the sphere, we can split so that the new points
    are equally distanced by angle (to the normalized points are equally distance)

    Args:
        edge_vertices:  extremitie of shape (E,2,3)
        num_segments: int,
        use_length: bool, set to false if the new points should be equally distanced
                    by angle
    Returns:
        Array of shape (E*(num_segments-1), 3)
    """
    if num_segments <= 1:
        return np.array([])
    t = np.arange(1, num_segments) / num_segments
    if use_angle:
        # shape (E,)
        omegas = np.arccos(np.sum(edge_extremes[:, 0] * edge_extremes[:, 1], axis=-1))
        sin_om = np.sin(omegas)[:, None]
        u = np.sin(omegas[:, None] * (1 - t[None, :]))
        v = np.sin(omegas[:, None] * t[None, :]) / sin_om
        u = u[:, :, None]  # shape (E,num_segments-1, 1)
        v = v[:, :, None]  # shape (E,num_segments-1, 1)
    else:
        v = t[None, :, None]  # interpolation weights, shape (1, num_segments-1, 1)
        u = 1 - v

    vertices_on_edges = u * edge_extremes[:, [0]]  # shape (E,ns,3)
    vertices_on_edges = vertices_on_edges + v * edge_extremes[:, [1]]  # shape(E,ns-1,3)
    vertices_on_edges = vertices_on_edges.reshape(-1, 3)
    return vertices_on_edges


def triangle_refine(
    vertices: npt.NDArray,
    faces: npt.NDArray,
    factor: int,
    use_angle: bool = False,
    normalize: bool = True,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Adapted from https://github.com/vedranaa/icosphere
    Given a base mesh, refine it using 1/depth factor

    Args:
        vertices (array): [TODO:description]
        faces (array): [TODO:description]
        depth (array): [TODO:description]

    Returns:
        Vertices and faces of the refined mesh
    """
    if factor <= 1:
        return vertices, faces
    # shape [E, 3], where E = F * 3
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0
    )

    # sort in alphabetic order and remove duplicates
    # WARN: shape (2,E)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    num_faces = faces.shape[0]
    num_vertices = vertices.shape[0]
    num_edges = edges.shape[0]
    subfaces = np.empty((num_faces * factor**2, 3), dtype=int)
    new_vertices = np.empty(
        (
            num_vertices
            + num_edges * (factor - 1)  # Vertices on edges
            + num_faces
            * (factor - 1)
            * (factor - 2)
            // 2,  # vertices on faces, none if d=2
            3,
        )
    )
    new_vertices[:num_vertices] = vertices

    # Dictionary used to determine the direction of the edge to avoid redundancy
    edge_index = dict()
    for i in range(num_edges):
        edge_index[(edges[i, 0], edges[i, 1])] = i
        edge_index[(edges[i, 1], edges[i, 0])] = i

    template = triangle_template(factor)
    ordering = triangles_order(factor)
    reordered_template = ordering[template]

    edge_extremes = vertices[edges]  # shape (E,2,3)
    vertices_on_edges = split_edges(
        edge_extremes,
        num_segments=factor,
        use_angle=use_angle,
    )
    # Step 1: e add (depth-1) vertices per edge
    new_vertices[num_vertices : num_vertices + num_edges * (factor - 1)] = (
        vertices_on_edges
    )

    r = np.arange(factor - 1) + num_vertices
    start_idx = num_edges * (factor - 1) + num_vertices
    num_inside_vertices = (factor - 1) * (factor - 2) // 2
    T_ref = np.arange(start_idx, num_inside_vertices + start_idx)
    for f in range(num_faces):
        # First, fixing connectivity. We get hold of the indices of all
        # vertices invoved in this subface: original, on-edges and on-faces.
        T = T_ref + f * num_inside_vertices
        eAB = (faces[f, 0], faces[f, 1])
        eAC = (faces[f, 0], faces[f, 2])
        eBC = (faces[f, 1], faces[f, 2])
        # -- Already added in Step 1
        # Sorting the vertices on edges in the right order
        # making sure edge is oriented from lower to higher vertex index
        AB = (edge_index[eAB] * (factor - 1) + r)[:: e_direction(eAB)]
        AC = (edge_index[eAC] * (factor - 1) + r)[:: e_direction(eAC)]
        BC = (edge_index[eBC] * (factor - 1) + r)[:: e_direction(eBC)]
        VEF = np.r_[faces[f], AB, AC, BC, T]  # vertices in template order
        # sort vertices in ordering
        subfaces[f * factor**2 : (f + 1) * factor**2, :] = VEF[reordered_template]
        # Now geometry, computing positions of on face vertices.
        new_vertices[T, :] = triangle_interior(
            new_vertices[AB, :],
            new_vertices[AC, :],
            use_angle=use_angle,
        )

    # normalize vertices
    if normalize:
        new_vertices = new_vertices / np.linalg.norm(
            new_vertices, axis=1, keepdims=True
        )

    return (new_vertices, subfaces)


def triangle_template(nu: int) -> npt.NDArray[np.int64]:
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
            faces.append([j + vertex0, j + vertex0 + skip, j + vertex0 + skip + 1])
            faces.append([j + vertex0, j + vertex0 + skip + 1, j + vertex0 + 1])
        # adding the last (unpaired, rightmost) triangle
        faces.append([i + vertex0, i + vertex0 + skip, i + vertex0 + skip + 1])

    return np.array(faces)


def triangles_order(nu: int):
    """
    Permutation for ordering of                 0
    face vertices which transformes            / \
    reading-order indexing into indexing      3---6
    first corners vertices, then on-edges    / \\/ \
    vertices, and then on-face vertices     4---12--7
    (as illustrated).                      / \\/ \\/ \
                                          5---13--14--8
                                         / \\/ \\/ \\/ \\
                                        1---9--10--11---2
    Args:
        nu (int): depth for which to generate the ordering

    Returns:
        return ordering of length $(nu+1)(nu+2)/2$
    """

    left = [j for j in range(3, nu + 2)]
    right = [j for j in range(nu + 2, 2 * nu + 1)]
    bottom = [j for j in range(2 * nu + 1, 3 * nu)]
    inside = [j for j in range(3 * nu, (nu + 1) * (nu + 2) // 2)]

    o = [0]  # topmost corner
    for i in range(nu - 1):
        o.append(left[i])
        o = o + inside[i * (i - 1) // 2 : i * (i + 1) // 2]
        o.append(right[i])
    o = o + [1] + bottom + [2]

    return np.array(o)


def triangle_interior(AB: npt.NDArray, AC: npt.NDArray, use_angle: bool = False):
    """
    Returns coordinates of the inside (on-face) vertices (marked by star) for subdivision
    of the face ABC when given coordinates of the on-edge verticesAB[i] and AC[i].                     
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

    vAB: ndarray, shape(depth-2,3)
    vAC: ndarray, shape(depth-2,3)
    Returns:
        [TODO:return]
    """
    if AB.shape[0] <= 1:
        return None
    vertices = []
    for i in range(1, AB.shape[0]):
        vertices.append(
            split_edges(
                np.concatenate([AB[None, [i]], AC[None, [i]]], axis=1),
                i + 1,
                use_angle,
            )
        )
    all_vertices = np.concatenate(vertices, axis=0)
    return all_vertices


def square_refine(
    vertices: npt.NDArray,
    squares: npt.NDArray,
    factor: int,
    use_length: bool = True,
    normalize: bool = True,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Adapted from https://github.com/vedranaa/icosphere
    Given a base mesh, refine it using 1/depth factor

    Args:
        vertices (array): [TODO:description]
        faces (array): [TODO:description]
        depth (array): [TODO:description]

    Returns:
        Vertices and faces of the refined mesh
    """
    if factor <= 1:
        return vertices, squares
    # shape [E, 3], where E = F * 3
    edges = np.concatenate(
        [
            squares[:, [0, 1]],
            squares[:, [1, 2]],
            squares[:, [2, 3]],
            squares[:, [3, 0]],
        ],
        axis=0,
    )

    # sort in alphabetic order and remove duplicates
    # shape (E,2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    num_squares = squares.shape[0]
    num_vertices = vertices.shape[0]
    num_edges = edges.shape[0]
    subsquares = np.empty((num_squares * factor**2, 4), dtype=int)
    new_vertices = np.empty(
        (
            num_vertices
            + num_edges * (factor - 1)  # Vertices on edges
            + num_squares * (factor - 1) ** 2,  # vertices on faces, 1 if factor=2
            3,
        )
    )
    new_vertices[:num_vertices] = vertices

    # Dictionary used to determine the direction of the edge to avoid redundancy
    edge_index = dict()
    for i in range(num_edges):
        edge_index[tuple(edges[i])] = i
        edge_index[tuple(edges[i][::-1])] = i

    template = square_template(factor)
    ordering = squares_order(factor)
    reordered_template = ordering[template]

    edge_extremes = vertices[edges]  # shape (E,2,3)
    vertices_on_edges = split_edges(
        edge_extremes,
        num_segments=factor,
        use_angle=use_length,
    )
    # Step 1: e add (depth-1) vertices per edge
    new_vertices[num_vertices : num_vertices + num_edges * (factor - 1)] = (
        vertices_on_edges
    )
    r = np.arange(factor - 1) + num_vertices
    start_idx = num_edges * (factor - 1) + num_vertices
    num_inside_vertices = (factor - 1) ** 2
    # this will be offset later as indices for the vertices on edges
    T_ref = np.arange(start_idx, num_inside_vertices + start_idx)
    for f in range(num_squares):
        # First, fixing connectivity. We get hold of the indices of all
        # vertices invoved in this subface: original, on-edges and on-faces.
        # T containes the indices of on-faces vertices
        T = T_ref + f * num_inside_vertices
        eAB = squares[f, [0, 1]]
        eBC = squares[f, [1, 2]]
        eCD = squares[f, [2, 3]]
        eAD = squares[f, [0, 3]]
        # -- Already added in Step 1
        # Sorting the vertices on edges in the right order
        AB = (edge_index[tuple(eAB)] * (factor - 1) + r)[:: e_direction(eAB)]
        BC = (edge_index[tuple(eBC)] * (factor - 1) + r)[:: e_direction(eBC)]
        CD = (edge_index[tuple(eCD)] * (factor - 1) + r)[:: e_direction(eCD)]
        AD = (edge_index[tuple(eAD)] * (factor - 1) + r)[:: e_direction(eAD)]
        # --
        VEF = np.r_[squares[f], AB, AD, BC, CD, T]
        subsquares[f * factor**2 : (f + 1) * factor**2, :] = VEF[reordered_template]
        # Now geometry, computing positions of face vertices.
        new_vertices[T, :] = square_interior(
            new_vertices[AD, :],
            new_vertices[BC, :],
            use_length=use_length,
        )
    # normalize vertices
    if normalize:
        new_vertices = new_vertices / np.linalg.norm(
            new_vertices,
            axis=1,
            keepdims=True,
        )

    return (new_vertices, subsquares)


def square_template(nu: int) -> npt.NDArray[np.int64]:
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
    face vertices which transformes       |  |   |   |
    reading-order indexing into indexing  6==12==13==8
    first corners vertices, then on-edges |  |   |   |
    vertices, and then on-face vertices   7==14==15==9
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


def square_interior(AD: npt.NDArray, BC: npt.NDArray, use_length: bool = True):
    """
     Returns coordinates of the inside (on-face) vertices (marked by star) for subdivision
     of the face ABCC when given coordinates of the on-edge vertices AD[i] and BC[i].
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
         [TODO:return]
    """
    assert AD.shape[0] == BC.shape[0]
    extremities = np.concatenate([AD[:, None], BC[:, None]], axis=1)
    return split_edges(
        extremities,
        num_segments=AD.shape[0] + 1,
        use_angle=use_length,
    )


def e_direction(edge) -> Literal[-1, 1]:
    """
    Parameters
    ----------
    edge : Tuple, containing the indices of the edge vertices

    Returns
    -------
        -1 if edge[0]>edges[1] and 1 otherwise
    """
    return 1 - 2 * (edge[0] > edge[1])


def compute_angles_per_depth(max_depth=100):
    phi = (1 + np.sqrt(5)) / 2
    initial_vertices = np.array([[-1, -phi, 0], [1, -phi, 0]])
    initial_vertices = initial_vertices / np.linalg.norm(
        initial_vertices, axis=1, keepdims=True
    )
    angles = []
    for d in range(1, max_depth + 1):
        left_vertex = initial_vertices[0]
        right_vertex = left_vertex + (initial_vertices[1] - initial_vertices[0]) / d
        vertices = np.array([left_vertex, right_vertex])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        angles.append(np.arccos(np.inner(vertices[0], vertices[1])) / np.pi * 180)
    return np.array(angles)


def compute_edges_lenghts(
    vertices: npt.NDArray,
    edges: npt.NDArray[np.int_],
) -> npt.NDArray:
    """Given the vertices and the edges, compute the length of each edge

    Args:
        vertices (numpy array): shape (K,3)
        edges (numpy array): shape (2,E)
    Returns:
        Lengths of the edges, shape (E,)
    """
    # edges: shape (2,E)
    edges_vertices = vertices[edges]  # shape (2, E, 3)
    edges_diff = edges_vertices[1] - edges_vertices[0]
    edges_lengths = np.linalg.norm(edges_diff, axis=-1)  # shape (E,)
    return edges_lengths


def compute_edges_angles(
    vertices: npt.NDArray,
    faces: npt.NDArray,
) -> npt.NDArray:
    """
    Calculate the angles between vertices based on the lengths of edges.

    Parameters:
    vertices (npt.NDArray): An array of vertex coordinates.
    faces (npt.NDArray): An array of face indices that define the connectivity of vertices.

    Returns:
    npt.NDArray: An array of angles (in degrees) between vertices calculated from edge lengths.
    """
    edges_lengths = compute_edges_lenghts(vertices, faces)
    angles_between_vertices = 360 * np.arcsin(edges_lengths / 2) / np.pi
    return angles_between_vertices


def change_grid(
    source_xyz: npt.NDArray,
    target_xyz: npt.NDArray,
    n_neighbors: int = -1,
    radius: float = -1.0,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Transpose a grid by finding the nearest neighbors of target points
    among the source points based on geographic coordinates.

    Parameters:
    source_latlon (npt.NDArray): An array of shape (S, 2) containing
                                  longitude and latitude of source points.
    target_latlon (npt.NDArray): An array of shape (T, 2) containing
                                  longitude and latitude of target points.
    n_neighbors (int, optional): The number of nearest neighbors to find.
                                  Must be negative if radius is specified.
                                  Default is -1.
    radius (float, optional): The radius within which to search for neighbors.
                              Must be negative if n_neighbors is specified.
                              Default is -1.0.

    Returns:
    Tuple[npt.NDArray, npt.NDArray]: A tuple containing:
        - nearest_indices (npt.NDArray): Indices of the nearest neighbors in
                                          the source points, shape (T,nearest_indices)
        - nearest_distances (npt.NDArray): Distances to the nearest neighbors,
        shape (T,nearest_indices)
    """
    # either n_neighbors or radius should be used but not both
    assert n_neighbors * radius < 0
    if radius > 0:
        mode = "radius"
        n_neighbors = 5
    else:
        mode = "count"
        radius = 1.0

    neigh = NearestNeighbors(
        n_neighbors=n_neighbors, metric="cosine", radius=radius
    ).fit(source_xyz)

    if mode == "count":
        nearest_distances, nearest_indices = neigh.kneighbors(target_xyz)
    else:
        nearest_distances, nearest_indices = neigh.radius_neighbors(target_xyz)

    return (nearest_indices, nearest_distances)
