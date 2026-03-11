# License: Non-Commercial Use Only
#
# Permission is granted to use, copy, modify, and distribute this software
# for non-commercial purposes only, with attribution to the original author.
# Commercial use requires explicit permission.
#
# This software is provided "as is", without warranty of any kind.

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
        u = np.sin(omegas[:, None] * (1 - t[None, :])) / sin_om
        v = np.sin(omegas[:, None] * t[None, :]) / sin_om
        u = u[:, :, None]  # shape (E,num_segments-1, 1)
        v = v[:, :, None]  # shape (E,num_segments-1, 1)
    else:
        # interpolation weights, shape (1, num_segments-1, 1)
        v = t[None, :, None]
        u = 1 - v
    # shape (E,ns-1,3)
    nodes_on_edges = u * edge_extremes[:, 0][:, None]
    # shape (E,ns-1,3)
    nodes_on_edges = nodes_on_edges + v * edge_extremes[:, 1][:, None]
    nodes_on_edges = nodes_on_edges.reshape(-1, 3)
    return nodes_on_edges


def refine_triangles(
    nodes: NDArray,
    triangles: NDArray,
    factor: int,
    use_angle: bool = False,
) -> Tuple[NDArray, NDArray]:
    """Refine a triangular mesh by subdividing each face.

    Vectorized implementation — all edge splitting, interior computation,
    and connectivity wiring are batched into array operations with no
    per-face Python loop.

    Args:
        nodes: Coordinates of the mesh nodes, shape (N, 3).
        triangles: Triangular face indices, shape (T, 3).
        factor: Subdivision factor per edge.
        use_angle: Use angle-based (geodesic) interpolation.

    Returns:
        Tuple of (refined_nodes, refined_faces) where refined_nodes has
        shape (M, 3) and refined_faces has shape (T*factor^2, 3).
    """
    if factor <= 1:
        return nodes, triangles

    nu = factor
    n_tri = triangles.shape[0]
    n_nodes = nodes.shape[0]

    # --- Unique edges ---
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]],
        axis=0,
    )
    edges_sorted = np.sort(edges, axis=1)
    edges_unique, inverse = np.unique(
        edges_sorted, axis=0, return_inverse=True
    )
    n_edges = edges_unique.shape[0]
    # inverse maps each of the 3*n_tri directed edges to its unique index
    # Split into per-edge-type arrays: ab, bc, ac  (each length n_tri)
    inv_ab = inverse[:n_tri]
    inv_bc = inverse[n_tri : 2 * n_tri]
    inv_ac = inverse[2 * n_tri :]

    # --- Node counts ---
    ns = nu - 1  # new nodes per edge
    n_interior = (nu - 1) * (nu - 2) // 2  # interior nodes per face
    total_new = n_edges * ns + n_tri * n_interior

    new_nodes = np.empty((n_nodes + total_new, 3))
    new_nodes[:n_nodes] = nodes

    # --- Step 1: split all edges in one batch call ---
    edge_extremes = nodes[edges_unique]  # (E, 2, 3)
    edge_node_coords = split_edges(edge_extremes, nu, use_angle)
    # shape (E*ns, 3) — stored as E blocks of ns nodes each
    new_nodes[n_nodes : n_nodes + n_edges * ns] = edge_node_coords

    # --- Step 2: build per-face edge-node indices (vectorized) ---
    # For each triangle's 3 edges, compute the global indices of the ns
    # on-edge nodes, respecting direction.
    #
    # edge_base[e] = n_nodes + e * ns  gives the start of edge e's nodes.
    # direction: if triangle edge (a,b) has a < b, nodes go forward;
    #            if a > b, nodes go in reverse.
    edge_base = n_nodes + np.arange(n_edges) * ns  # (E,)
    r = np.arange(ns)  # (ns,)

    # For each face edge, the unsorted indices are edge_base[inv] + r
    # shape: (n_tri, ns)
    ab_idx = edge_base[inv_ab, None] + r[None, :]
    ac_idx = edge_base[inv_ac, None] + r[None, :]
    bc_idx = edge_base[inv_bc, None] + r[None, :]

    # Reverse where the triangle edge is "backwards" (a > b for that edge)
    ab_flip = triangles[:, 0] > triangles[:, 1]
    ac_flip = triangles[:, 0] > triangles[:, 2]
    bc_flip = triangles[:, 1] > triangles[:, 2]

    ab_idx[ab_flip] = ab_idx[ab_flip, ::-1]
    ac_idx[ac_flip] = ac_idx[ac_flip, ::-1]
    bc_idx[bc_flip] = bc_idx[bc_flip, ::-1]

    # --- Step 3: interior nodes (vectorized across faces) ---
    interior_start = n_nodes + n_edges * ns
    if n_interior > 0:
        # For each face f, interior nodes occupy indices
        # interior_start + f*n_interior ... interior_start + (f+1)*n_interior
        #
        # Interior row i (1 <= i <= ns-1) has i points interpolated between
        # ab[i] and ac[i]. We precompute all (row_index, t_value) pairs and
        # do a single batched interpolation.
        #
        # Total interior points per face = sum(1..ns-1) = n_interior
        # Build arrays:  row_i[k] = which row (1..ns-1) this point belongs to
        #                t_val[k] = interpolation parameter in (0, 1)
        row_ids = []
        t_vals = []
        for i in range(1, ns):
            t = np.arange(1, i + 1) / (i + 1)  # i values in (0,1)
            row_ids.append(np.full(i, i, dtype=np.intp))
            t_vals.append(t)
        row_ids = np.concatenate(row_ids)   # (n_interior,)
        t_vals = np.concatenate(t_vals)     # (n_interior,)

        # Gather the endpoint coords for all faces at once
        # ab_idx[:, row_ids] -> (n_tri, n_interior) indices into new_nodes
        ab_pts = new_nodes[ab_idx[:, row_ids]]  # (n_tri, n_interior, 3)
        ac_pts = new_nodes[ac_idx[:, row_ids]]  # (n_tri, n_interior, 3)

        # Interpolate: shape (n_interior,) broadcast over (n_tri, n_interior, 3)
        t_b = t_vals[None, :, None]  # (1, n_interior, 1)
        if use_angle:
            dots = np.sum(ab_pts * ac_pts, axis=-1, keepdims=True)
            omegas = np.arccos(np.clip(dots, -1, 1))  # (n_tri, n_interior, 1)
            sin_om = np.sin(omegas)
            u = np.sin(omegas * (1 - t_b)) / sin_om
            v = np.sin(omegas * t_b) / sin_om
            face_interior = u * ab_pts + v * ac_pts
        else:
            face_interior = (1 - t_b) * ab_pts + t_b * ac_pts

        new_nodes[interior_start : interior_start + n_tri * n_interior] = (
            face_interior.reshape(-1, 3)
        )

    # --- Step 4: connectivity template (vectorized) ---
    template = triangle_template(nu)
    ordering = triangles_order(nu)
    reordered_template = ordering[template]  # (nu^2, 3)

    # Build the sorted_nodes array for ALL faces at once: (n_tri, n_verts)
    # where n_verts = 3 + 3*ns + n_interior
    # Order: [corner0, corner1, corner2, ab_nodes, ac_nodes, bc_nodes, interior]
    face_interior_idx = (
        interior_start
        + np.arange(n_tri)[:, None] * n_interior
        + np.arange(n_interior)[None, :]
    )  # (n_tri, n_interior)

    sorted_nodes = np.concatenate(
        [triangles, ab_idx, ac_idx, bc_idx, face_interior_idx], axis=1
    )  # (n_tri, n_verts)

    # Apply template to all faces at once
    sub_triangles = sorted_nodes[:, reordered_template.ravel()].reshape(
        n_tri * nu**2, 3
    )

    # --- Normalize ---
    new_nodes /= np.linalg.norm(new_nodes, axis=1, keepdims=True)

    return new_nodes, sub_triangles


def _refine_triangles_reference(
    nodes: NDArray,
    triangles: NDArray,
    factor: int,
    use_angle: bool = False,
) -> Tuple[NDArray, NDArray]:
    """Reference (non-vectorized) implementation of refine_triangles.

    Kept for correctness validation. See :func:`refine_triangles` for the
    optimized version.
    """
    if factor <= 1:
        return nodes, triangles
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [0, 2]]],
        axis=0,
    )

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
                    n_edges * (factor - 1)
                    + n_triangles
                    * (factor - 1)
                    * (factor - 2)
                    // 2,
                    3,
                )
            ),
        ],
        axis=0,
    )

    edge_index = {}
    for i in range(n_edges):
        edge_index[(edges[i, 0], edges[i, 1])] = i
        edge_index[(edges[i, 1], edges[i, 0])] = i

    template = triangle_template(factor)
    ordering = triangles_order(factor)
    reordered_template = ordering[template]

    edge_extremes = nodes[edges]
    nodes_on_edges = split_edges(
        edge_extremes,
        num_segments=factor,
        use_angle=use_angle,
    )
    new_nodes[n_nodes : n_nodes + n_edges * (factor - 1)] = nodes_on_edges

    r = np.arange(factor - 1) + n_nodes
    start_idx = n_edges * (factor - 1) + n_nodes
    num_inside_nodes = (factor - 1) * (factor - 2) // 2
    ref_triangle = np.arange(start_idx, num_inside_nodes + start_idx)

    for triangle_idx in range(n_triangles):
        triangle_nodes = ref_triangle + triangle_idx * num_inside_nodes
        triangle = triangles[triangle_idx]
        e_ab = (triangle[0], triangle[1])
        e_ac = (triangle[0], triangle[2])
        e_bc = (triangle[1], triangle[2])
        sorted_ab = (edge_index[e_ab] * (factor - 1) + r)[:: e_direction(e_ab)]
        sorted_ac = (edge_index[e_ac] * (factor - 1) + r)[:: e_direction(e_ac)]
        sorted_bc = (edge_index[e_bc] * (factor - 1) + r)[:: e_direction(e_bc)]
        sorted_nodes = np.r_[
            triangle,
            sorted_ab,
            sorted_ac,
            sorted_bc,
            triangle_nodes,
        ]
        sub_triangles[
            triangle_idx * factor**2 : (triangle_idx + 1) * factor**2, :
        ] = sorted_nodes[reordered_template]
        new_nodes[triangle_nodes, :] = triangle_interior(
            new_nodes[sorted_ab, :],
            new_nodes[sorted_ac, :],
            use_angle=use_angle,
        )

    new_nodes = new_nodes / np.linalg.norm(new_nodes, axis=1, keepdims=True)

    return (new_nodes, sub_triangles)


def triangle_template(nu: int) -> NDArray[np.int64]:
    r"""Template for linking subfaces in a subdivision of a triangular face.

    Returns faces with node indexing given by reading order.

    Illustration for nu=4::

                 0
                / \
               1---2
              / \ / \
             3---4---5
            / \ / \ / \
           6---7---8---9
          / \ / \ / \ / \
         10--11--12--13--14

    Args:
        nu: Subdivision factor.

    Returns:
        Faces template of shape (nu^2, 3).
    """
    # Build all upward and downward triangles using vectorized index math.
    # Row i of the triangle grid has nodes node0..node0+i where
    # node0 = i*(i+1)//2.  Row i has i upward + i downward + 1 tip triangle.
    rows = np.arange(nu)
    node0 = rows * (rows + 1) // 2  # start index of each row

    # Pre-allocate
    faces = np.empty((nu**2, 3), dtype=np.intp)
    idx = 0
    for i in range(nu):
        n0 = node0[i]
        s = i + 1
        if i > 0:
            j = np.arange(i)
            # Upward triangles
            faces[idx : idx + i, 0] = j + n0
            faces[idx : idx + i, 1] = j + n0 + s
            faces[idx : idx + i, 2] = j + n0 + s + 1
            idx += i
            # Downward triangles
            faces[idx : idx + i, 0] = j + n0
            faces[idx : idx + i, 1] = j + n0 + s + 1
            faces[idx : idx + i, 2] = j + n0 + 1
            idx += i
        # Rightmost tip triangle
        faces[idx] = [i + n0, i + n0 + s, i + n0 + s + 1]
        idx += 1
    return faces


def triangles_order(nu: int):
    r"""Permutation that reorders reading-order node indices.

    Reorders into: corners first, then on-edge nodes, then interior nodes.

    Illustration for nu=4::

                 0
                / \
               3---6
              / \ / \
             4---12--7
            / \ / \ / \
           5---13--14--8
          / \ / \ / \ / \
         1---9--10--11---2

    Args:
        nu: Subdivision factor.

    Returns:
        Ordering array of length (nu+1)*(nu+2)//2.
    """
    n_verts = (nu + 1) * (nu + 2) // 2
    o = np.empty(n_verts, dtype=np.intp)

    left = np.arange(3, nu + 2)
    right = np.arange(nu + 2, 2 * nu + 1)
    bottom = np.arange(2 * nu + 1, 3 * nu)
    inside = np.arange(3 * nu, n_verts)

    pos = 0
    o[pos] = 0  # top corner
    pos += 1
    for i in range(nu - 1):
        o[pos] = left[i]
        pos += 1
        n_inside_row = i  # row i has i interior nodes
        if n_inside_row > 0:
            o[pos : pos + n_inside_row] = inside[
                i * (i - 1) // 2 : i * (i + 1) // 2
            ]
            pos += n_inside_row
        o[pos] = right[i]
        pos += 1
    # Bottom row: corner 1, bottom edges, corner 2
    o[pos] = 1
    pos += 1
    n_bottom = len(bottom)
    o[pos : pos + n_bottom] = bottom
    pos += n_bottom
    o[pos] = 2
    return o


def triangle_interior(ab: NDArray, ac: NDArray, use_angle: bool = False):
    r"""Compute interior nodes for a subdivided triangular face.

    Given on-edge nodes AB[i] and AC[i], computes the interior nodes
    (marked by ``*`` below)::

              A
             / \
           AB0--AC0
           / \ / \
         AB1--*--AC1
         / \ / \ / \
       AB2---*---*---AC2
        / \ / \ / \ / \
       B-BC1--BC2--BC3-C

    Args:
        ab: On-edge nodes from A toward B, shape (factor-1, 3).
        ac: On-edge nodes from A toward C, shape (factor-1, 3).
        use_angle: Use angle-based (geodesic) interpolation between
            corresponding AB and AC nodes.

    Returns:
        Interior node coordinates, shape ((factor-1)*(factor-2)//2, 3).
        Returns None when factor <= 2 (no interior nodes).
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
    use_angle: bool = False,
) -> Tuple[NDArray, NDArray]:
    """Refine a rectangular mesh by subdividing each face.

    Vectorized implementation — all edge splitting, interior computation,
    and connectivity wiring are batched into array operations with no
    per-face Python loop.

    Args:
        nodes: Coordinates of the mesh nodes, shape (N, 3).
        rectangles: Rectangular face indices, shape (R, 4).
        factor: Subdivision factor per edge.
        use_angle: Use angle-based (geodesic) interpolation.

    Returns:
        Tuple of (refined_nodes, refined_faces) where refined_nodes has
        shape (M, 3) and refined_faces has shape (R*factor^2, 4).
    """
    if factor <= 1:
        return nodes, rectangles

    nu = factor
    ns = nu - 1  # new nodes per edge
    n_rect = rectangles.shape[0]
    n_nodes = nodes.shape[0]

    # --- Unique edges ---
    # 4 edges per rect: AB(0-1), BC(1-2), CD(2-3), AD(0-3)
    all_edges = np.concatenate(
        [
            rectangles[:, [0, 1]],
            rectangles[:, [1, 2]],
            rectangles[:, [2, 3]],
            rectangles[:, [0, 3]],
        ],
        axis=0,
    )
    edges_sorted = np.sort(all_edges, axis=1)
    edges_unique, inverse = np.unique(
        edges_sorted, axis=0, return_inverse=True
    )
    n_edges = edges_unique.shape[0]
    # Split inverse into per-edge-type arrays (each length n_rect)
    inv_ab = inverse[:n_rect]
    inv_bc = inverse[n_rect : 2 * n_rect]
    inv_cd = inverse[2 * n_rect : 3 * n_rect]
    inv_ad = inverse[3 * n_rect :]

    # --- Node counts ---
    n_interior = ns**2  # interior nodes per face
    total_new = n_edges * ns + n_rect * n_interior

    new_nodes = np.empty((n_nodes + total_new, 3))
    new_nodes[:n_nodes] = nodes

    # --- Step 1: split all edges in one batch call ---
    edge_extremes = nodes[edges_unique]  # (E, 2, 3)
    new_nodes[n_nodes : n_nodes + n_edges * ns] = split_edges(
        edge_extremes, nu, use_angle
    )

    # --- Step 2: per-face edge-node indices (vectorized) ---
    edge_base = n_nodes + np.arange(n_edges) * ns  # (E,)
    r = np.arange(ns)  # (ns,)

    # shape: (n_rect, ns)
    ab_idx = edge_base[inv_ab, None] + r[None, :]
    bc_idx = edge_base[inv_bc, None] + r[None, :]
    cd_idx = edge_base[inv_cd, None] + r[None, :]
    ad_idx = edge_base[inv_ad, None] + r[None, :]

    # Reverse where the rect edge is "backwards" (a > b)
    ab_flip = rectangles[:, 0] > rectangles[:, 1]
    bc_flip = rectangles[:, 1] > rectangles[:, 2]
    cd_flip = rectangles[:, 2] > rectangles[:, 3]
    ad_flip = rectangles[:, 0] > rectangles[:, 3]

    ab_idx[ab_flip] = ab_idx[ab_flip, ::-1]
    bc_idx[bc_flip] = bc_idx[bc_flip, ::-1]
    cd_idx[cd_flip] = cd_idx[cd_flip, ::-1]
    ad_idx[ad_flip] = ad_idx[ad_flip, ::-1]

    # --- Step 3: interior nodes (vectorized) ---
    # Rectangle interior: for each row i (0..ns-1), interpolate ns points
    # between AD[i] and BC[i]. All rows use the same num_segments = ns+1.
    # So we can batch ALL rows of ALL faces into one split_edges call.
    interior_start = n_nodes + n_edges * ns
    if n_interior > 0:
        # ad_idx[:, i] and bc_idx[:, i] are the endpoints for row i
        # We need all (n_rect * ns) pairs, each split into ns+1 segments.
        # ad_idx shape: (n_rect, ns), bc_idx shape: (n_rect, ns)
        ad_pts = new_nodes[ad_idx]  # (n_rect, ns, 3)
        bc_pts = new_nodes[bc_idx]  # (n_rect, ns, 3)

        # Reshape to (n_rect * ns, 2, 3) for a single split_edges call
        pairs = np.stack(
            [ad_pts.reshape(-1, 3), bc_pts.reshape(-1, 3)], axis=1
        )
        interior_flat = split_edges(pairs, ns + 1, use_angle)
        # shape: (n_rect * ns * ns, 3)
        # Reshape to (n_rect, ns, ns, 3) -> face-major order
        interior = interior_flat.reshape(n_rect, ns, ns, 3)
        # Flatten interior dims: (n_rect, ns^2, 3)
        interior = interior.reshape(n_rect, n_interior, 3)

        new_nodes[interior_start : interior_start + n_rect * n_interior] = (
            interior.reshape(-1, 3)
        )

    # --- Step 4: connectivity template (vectorized) ---
    template = rectangle_template(nu)
    ordering = rectangles_order(nu)
    reordered_template = ordering[template]  # (nu^2, 4)

    # Build sorted_nodes for ALL faces: (n_rect, n_verts)
    # Order: [4 corners, ab_nodes, ad_nodes, bc_nodes, cd_nodes, interior]
    face_interior_idx = (
        interior_start
        + np.arange(n_rect)[:, None] * n_interior
        + np.arange(n_interior)[None, :]
    )  # (n_rect, n_interior)

    sorted_nodes = np.concatenate(
        [rectangles, ab_idx, ad_idx, bc_idx, cd_idx, face_interior_idx],
        axis=1,
    )  # (n_rect, n_verts)

    # Apply template to all faces at once
    subrects = sorted_nodes[:, reordered_template.ravel()].reshape(
        n_rect * nu**2, 4
    )

    # --- Normalize ---
    new_nodes /= np.linalg.norm(new_nodes, axis=1, keepdims=True)

    return new_nodes, subrects


def _refine_rectrangles_reference(
    nodes: NDArray,
    rectangles: NDArray,
    factor: int,
    use_angle: bool = False,
) -> Tuple[NDArray, NDArray]:
    """Reference (non-vectorized) implementation of refine_rectrangles.

    Kept for correctness validation. See :func:`refine_rectrangles` for the
    optimized version.
    """
    if factor <= 1:
        return nodes, rectangles
    edges = np.concatenate(
        [
            rectangles[:, [0, 1]],
            rectangles[:, [1, 2]],
            rectangles[:, [2, 3]],
            rectangles[:, [3, 0]],
        ],
        axis=0,
    )

    edges = np.unique(np.sort(edges, axis=1), axis=0)
    num_rects = rectangles.shape[0]
    num_nodes = nodes.shape[0]
    num_edges = edges.shape[0]
    subrects = np.empty((num_rects * factor**2, 4), dtype=int)
    new_nodes = np.empty(
        (
            num_nodes
            + num_edges * (factor - 1)
            + num_rects * (factor - 1) ** 2,
            3,
        )
    )
    new_nodes[:num_nodes] = nodes

    edge_index = {}
    for i in range(num_edges):
        edge_index[tuple(edges[i])] = i
        edge_index[tuple(edges[i][::-1])] = i

    template = rectangle_template(factor)
    ordering = rectangles_order(factor)
    reordered_template = ordering[template]

    edge_extremes = nodes[edges]
    new_nodes[num_nodes : num_nodes + num_edges * (factor - 1)] = split_edges(
        edge_extremes,
        num_segments=factor,
        use_angle=use_angle,
    )

    r = np.arange(factor - 1) + num_nodes
    start_idx = num_edges * (factor - 1) + num_nodes
    num_inside_nodes = (factor - 1) ** 2
    ref_rect = np.arange(start_idx, num_inside_nodes + start_idx)
    for f in range(num_rects):
        rect_nodes = ref_rect + f * num_inside_nodes
        rectangle = rectangles[f]
        e_ab = rectangle[[0, 1]]
        e_bc = rectangle[[1, 2]]
        e_cd = rectangle[[2, 3]]
        e_ad = rectangle[[0, 3]]
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
        sorted_nodes = np.r_[
            rectangles[f],
            sorted_ab,
            sorted_ad,
            sorted_bc,
            sorted_cd,
            rect_nodes,
        ]
        subrects[f * factor**2 : (f + 1) * factor**2, :] = sorted_nodes[
            reordered_template
        ]
        new_nodes[rect_nodes, :] = rectangle_interior(
            new_nodes[sorted_ad, :],
            new_nodes[sorted_bc, :],
            use_angle=use_angle,
        )
    new_nodes = new_nodes / np.linalg.norm(new_nodes, axis=1, keepdims=True)

    return (new_nodes, subrects)


def rectangle_template(nu: int) -> NDArray[np.int64]:
    r"""Template for linking subfaces in a subdivision of a rectangular face.

    Returns faces with node indexing given by reading order.

    Illustration for nu=3::

        0---1---2---3
        |   |   |   |
        4---5---6---7
        |   |   |   |
        8---9---10--11
        |   |   |   |
        12--13--14--15

    Args:
        nu: Subdivision factor.

    Returns:
        Faces template of shape (nu^2, 4).
    """

    i, j = np.meshgrid(np.arange(nu), np.arange(nu), indexing="ij")
    i = i.ravel()
    j = j.ravel()
    row0 = i * (nu + 1)
    row1 = (i + 1) * (nu + 1)
    faces = np.column_stack([row0 + j, row0 + j + 1, row1 + j + 1, row1 + j])
    return faces


def rectangles_order(nu: int):
    r"""Permutation that reorders reading-order node indices for rectangles.

    Reorders into: corners first, then on-edge nodes, then interior nodes.

    Illustration for nu=3::

        0---4---5---1
        |   |   |   |
        6---12--13--8
        |   |   |   |
        7---14--15--9
        |   |   |   |
        3---11--10--2

    Args:
        nu: Subdivision factor.

    Returns:
        Ordering array of length (nu+1)^2.
    """

    top = np.arange(4, nu + 3)
    left = np.arange(nu + 3, 2 * nu + 2)
    right = np.arange(2 * nu + 2, 3 * nu + 1)
    bottom = np.arange(3 * nu + 1, 4 * nu)[::-1]
    inside = np.arange(4 * nu, (nu + 1) ** 2).reshape(nu - 1, nu - 1)

    n_verts = (nu + 1) ** 2
    o = np.empty(n_verts, dtype=np.intp)
    pos = 0
    # Top row: corner 0, top edge, corner 1
    o[pos] = 0
    pos += 1
    o[pos : pos + len(top)] = top
    pos += len(top)
    o[pos] = 1
    pos += 1
    # Middle rows: left edge, interior, right edge
    for i in range(nu - 1):
        o[pos] = left[i]
        pos += 1
        o[pos : pos + nu - 1] = inside[i]
        pos += nu - 1
        o[pos] = right[i]
        pos += 1
    # Bottom row: corner 3, bottom edge (reversed), corner 2
    o[pos] = 3
    pos += 1
    o[pos : pos + len(bottom)] = bottom
    pos += len(bottom)
    o[pos] = 2
    return o


def rectangle_interior(ad: NDArray, bc: NDArray, use_angle: bool = False):
    """Compute interior nodes for a subdivided rectangular face.

    Given on-edge nodes AD[i] and BC[i], computes the interior nodes
    (marked by ``*`` below)::

        A---AB0--AB1--AB2--B
        |   |    |    |    |
       AD0--*----*----*---BC0
        |   |    |    |    |
       AD1--*----*----*---BC1
        |   |    |    |    |
       AD2--*----*----*---BC2
        |   |    |    |    |
        D---DC0--DC1--DC2--C

    Args:
        ad: On-edge nodes from A toward D, shape (factor-1, 3).
        bc: On-edge nodes from B toward C, shape (factor-1, 3).
        use_angle: Use angle-based (geodesic) interpolation.

    Returns:
        Interior node coordinates, shape ((factor-1)^2, 3).
    """
    assert ad.shape[0] == bc.shape[0]
    extremities = np.concatenate([ad[:, None], bc[:, None]], axis=1)
    return split_edges(
        extremities,
        num_segments=ad.shape[0] + 1,
        use_angle=use_angle,
    )


def e_direction(edge) -> Literal[-1, 1]:
    """Determines the direction of an edge based on its node indices.

    Args:
        edge: A tuple containing the indices of the edge nodes.
    Returns:
        -1 if edge[0] > edge[1], 1 otherwise.
    """
    return 1 - 2 * (edge[0] > edge[1])
