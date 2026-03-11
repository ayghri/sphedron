# License: Non-Commercial Use Only
#
# Permission is granted to use, copy, modify, and distribute this software
# for non-commercial purposes only, with attribution to the original author.
# Commercial use requires explicit permission.
#
# This software is provided "as is", without warranty of any kind.

from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from scipy import sparse
import trimesh

from .mesh.base import Mesh, RectangularMesh
from .helpers import query_nearest


# ---------------------------------------------------------------------------
# Private weight-computation helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(r: NDArray, kernel: str) -> NDArray:
    """Evaluate a conditionally positive definite RBF kernel.

    Args:
        r: Distance array (any shape).
        kernel: One of ``"thin_plate_spline"``, ``"cubic"``, or
            ``"linear"``.

    Returns:
        Kernel values, same shape as *r*.
    """
    if kernel == "thin_plate_spline":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0, r ** 2 * np.log(r), 0.0)
    if kernel == "cubic":
        return r ** 3
    if kernel == "linear":
        return r
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _knn_weights(
    distances: NDArray,
    indices: NDArray,
    method: str,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Compute sparse weight triplets from k-NN distances.

    Args:
        distances: Shape (M, k).
        indices: Shape (M, k).
        method: ``"nearest"``, ``"idw"``, or ``"gaussian"``.

    Returns:
        ``(row, col, val)`` arrays for sparse matrix construction.
    """
    M, k = indices.shape
    if method == "nearest":
        return (
            np.arange(M),
            indices[:, 0],
            np.ones(M),
        )

    if method == "idw":
        w = 1.0 / (distances + 1e-16)
    elif method == "gaussian":
        d_min = np.min(distances, axis=1, keepdims=True) + 1e-16
        w = np.exp(-0.2 * (distances / d_min) ** 2)
    else:
        raise ValueError(f"Unknown knn method: {method}")

    w /= w.sum(axis=1, keepdims=True)

    rows = np.repeat(np.arange(M), k)
    cols = indices.ravel()
    vals = w.ravel()
    return rows, cols, vals


def _barycentric_weights(
    sender_mesh: Mesh,
    receiver_nodes: NDArray,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Compute barycentric interpolation weights (degree=3).

    Projects each receiver node onto the nearest triangle of the sender
    mesh and computes barycentric coordinates.

    Args:
        sender_mesh: The sender mesh (any type — quads are triangulated).
        receiver_nodes: Query points, shape (M, 3).

    Returns:
        ``(row, col, val)`` arrays for sparse matrix construction.
    """
    sender_trimesh = sender_mesh.build_trimesh()
    closest_pts, _, tri_indices = trimesh.proximity.closest_point(
        sender_trimesh, receiver_nodes
    )

    # Triangle vertex indices (already in masked-node space)
    tri_verts = sender_trimesh.faces[tri_indices]  # (M, 3)
    A = sender_mesh.nodes[tri_verts[:, 0]]
    B = sender_mesh.nodes[tri_verts[:, 1]]
    C = sender_mesh.nodes[tri_verts[:, 2]]
    P = closest_pts

    # Barycentric coordinates via Cramer's rule
    v0 = B - A
    v1 = C - A
    v2 = P - A
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01 + 1e-30
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    bary = np.column_stack([u, v, w])
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(axis=1, keepdims=True)

    M = receiver_nodes.shape[0]
    rows = np.repeat(np.arange(M), 3)
    cols = tri_verts.ravel()
    vals = bary.ravel()
    return rows, cols, vals


def _bilinear_weights(
    sender_mesh: RectangularMesh,
    receiver_nodes: NDArray,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Compute bilinear interpolation weights on quad faces (degree=4).

    Projects each receiver node onto the nearest triangle, maps back to
    the containing quad face, then computes bilinear weights.

    Args:
        sender_mesh: A rectangular mesh.
        receiver_nodes: Query points, shape (M, 3).

    Returns:
        ``(row, col, val)`` arrays for sparse matrix construction.
    """
    sender_trimesh = sender_mesh.build_trimesh()
    closest_pts, _, tri_indices = trimesh.proximity.closest_point(
        sender_trimesh, receiver_nodes
    )

    # Map triangle → face, then get quad vertex indices
    face_indices = sender_mesh.triangle2face_index(tri_indices)
    quad_verts = sender_mesh.faces[face_indices]  # (M, 4)
    A = sender_mesh.nodes[quad_verts[:, 0]]  # corner 0
    B = sender_mesh.nodes[quad_verts[:, 1]]  # corner 1
    C = sender_mesh.nodes[quad_verts[:, 2]]  # corner 2
    D = sender_mesh.nodes[quad_verts[:, 3]]  # corner 3
    P = closest_pts

    # Project into local (u, v) coordinates on the quad.
    # For a general quad ABCD, we use least-squares projection:
    #   P ≈ (1-u)(1-v)*A + u(1-v)*B + uv*C + (1-u)v*D
    # Solve via iterative Newton or direct for near-planar quads.
    # For spherical meshes the quads are near-planar, so a direct
    # approach using the two edge directions works well.
    e_u = B - A  # u-direction edge
    e_v = D - A  # v-direction edge
    dp = P - A
    # Solve [e_u | e_v]^T @ [e_u | e_v] @ [u; v] = [e_u | e_v]^T @ dp
    g11 = np.sum(e_u * e_u, axis=1)
    g12 = np.sum(e_u * e_v, axis=1)
    g22 = np.sum(e_v * e_v, axis=1)
    b1 = np.sum(dp * e_u, axis=1)
    b2 = np.sum(dp * e_v, axis=1)
    det = g11 * g22 - g12 * g12 + 1e-30
    u = (g22 * b1 - g12 * b2) / det
    v = (g11 * b2 - g12 * b1) / det
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    # Bilinear weights: A=(1-u)(1-v), B=u(1-v), C=uv, D=(1-u)v
    w0 = (1 - u) * (1 - v)
    w1 = u * (1 - v)
    w2 = u * v
    w3 = (1 - u) * v
    weights = np.column_stack([w0, w1, w2, w3])
    weights /= weights.sum(axis=1, keepdims=True)

    M = receiver_nodes.shape[0]
    rows = np.repeat(np.arange(M), 4)
    cols = quad_verts.ravel()
    vals = weights.ravel()
    return rows, cols, vals


def _poly_basis(X: NDArray, degree: int) -> NDArray:
    """Build polynomial basis matrix up to given degree in 3D.

    For degree *d*, the basis contains all monomials
    ``x^a * y^b * z^c`` with ``a + b + c <= d``, ordered by total
    degree then lexicographically.

    Args:
        X: Coordinate array, shape ``(..., 3)``.
        degree: Maximum total polynomial degree (>= 0).

    Returns:
        Basis matrix, shape ``(..., m)`` where
        ``m = (d+1)(d+2)(d+3)/6``.
    """
    cols = [np.ones(X.shape[:-1] + (1,))]
    if degree >= 1:
        cols.append(X)
    if degree >= 2:
        x, y, z = X[..., 0:1], X[..., 1:2], X[..., 2:3]
        for d in range(2, degree + 1):
            # All monomials of total degree exactly d
            for a in range(d, -1, -1):
                for b in range(d - a, -1, -1):
                    c = d - a - b
                    cols.append(x ** a * y ** b * z ** c)
    return np.concatenate(cols, axis=-1)


def _local_rbf_weights(
    sender_nodes: NDArray,
    receiver_nodes: NDArray,
    distances: NDArray,
    indices: NDArray,
    kernel: str = "thin_plate_spline",
    degree: int = 1,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Compute local RBF interpolation weights (vectorized).

    For each receiver node, builds a local k×k kernel system using its
    k nearest sender neighbors and solves for interpolation weights.
    All M systems are solved in a single batched ``np.linalg.solve`` call.

    Args:
        sender_nodes: Sender node coordinates, shape (N, 3).
        receiver_nodes: Receiver node coordinates, shape (M, 3).
        distances: Pre-computed distances to k neighbors, shape (M, k).
        indices: Indices of k neighbors in sender_nodes, shape (M, k).
        kernel: RBF kernel name.
        degree: Polynomial augmentation degree (0=constant, 1=linear,
            2=quadratic, etc.).  ``None`` disables augmentation.

    Returns:
        ``(row, col, val)`` arrays for sparse matrix construction.
    """
    M, k = indices.shape

    # Gather local neighborhoods: (M, k, 3)
    X = sender_nodes[indices]

    # Pairwise distances within each neighborhood: (M, k, k)
    # Using ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b to avoid (M,k,k,3) tensor
    XX = np.sum(X ** 2, axis=-1)  # (M, k)
    dots = np.einsum("mij,mkj->mik", X, X)  # (M, k, k)
    dist_sq = XX[:, :, None] + XX[:, None, :] - 2 * dots
    dist_sq = np.maximum(dist_sq, 0)
    Phi = _rbf_kernel(np.sqrt(dist_sq), kernel)

    # Evaluation distances: already have (M, k)
    phi = _rbf_kernel(distances, kernel)

    # Minimal regularization to prevent exact singularity.
    # The post-hoc weight check (phase 3 below) handles
    # ill-conditioned systems by falling back to degree=0.
    Phi[:, range(k), range(k)] += 1e-12

    def _build_augmented(Phi_sub, phi_sub, P_sub, p_q_sub):
        """Build and solve augmented RBF system [Phi P; P^T 0]."""
        M_s, k_s = phi_sub.shape
        m_s = P_sub.shape[2]
        n_s = k_s + m_s
        A = np.zeros((M_s, n_s, n_s))
        A[:, :k_s, :k_s] = Phi_sub
        A[:, :k_s, k_s:] = P_sub
        A[:, k_s:, :k_s] = P_sub.transpose(0, 2, 1)
        rhs = np.zeros((M_s, n_s, 1))
        rhs[:, :k_s, 0] = phi_sub
        rhs[:, k_s:, 0] = p_q_sub
        return np.linalg.solve(A, rhs)[:, :k_s, 0]

    if degree is not None and degree >= 0:
        # Polynomial augmentation for conditionally positive definite
        # kernels (TPS, cubic, linear, etc.).
        # Center coordinates around the local mean to improve
        # conditioning (on the unit sphere, raw xyz values ~1 but
        # local variations ~0.01, causing near-singularity).
        center = X.mean(axis=1, keepdims=True)  # (M, 1, 3)
        X_c = X - center                         # (M, k, 3)
        q_c = receiver_nodes - center[:, 0, :]    # (M, 3)

        P = _poly_basis(X_c, degree)              # (M, k, m)
        p_q = _poly_basis(q_c[:, None, :], degree)[:, 0, :]  # (M, m)

        if degree == 0:
            w = _build_augmented(Phi, phi, P, p_q)
        else:
            # Phase 1: SVD pre-filter for rank-deficient P
            sv = np.linalg.svd(P, compute_uv=False)
            bad = sv[:, -1] < 1e-10 * sv[:, 0]
            attempt = ~bad

            # Phase 2: solve for rows passing SVD check
            w = np.empty((M, k))
            if attempt.any():
                w[attempt] = _build_augmented(
                    Phi[attempt], phi[attempt], P[attempt], p_q[attempt])

            # Phase 3: post-solve safety net — catch remaining
            # instabilities not detected by SVD on P alone
            if attempt.any():
                max_abs = np.max(np.abs(w[attempt]), axis=1)
                solve_bad = max_abs > 2.0
                if solve_bad.any():
                    bad[np.where(attempt)[0][solve_bad]] = True

            # Phase 4: fallback to degree=0 for all bad rows
            if bad.any():
                P0 = np.ones((bad.sum(), k, 1))
                p_q0 = np.ones((bad.sum(), 1))
                w[bad] = _build_augmented(
                    Phi[bad], phi[bad], P0, p_q0)
    else:
        w = np.linalg.solve(Phi, phi[:, :, None])[:, :, 0]

    rows = np.repeat(np.arange(M), k)
    cols = indices.ravel()
    vals = w.ravel()
    return rows, cols, vals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poly_terms(degree: int, dim: int) -> int:
    """Number of polynomial terms up to *degree* in *dim* dimensions.

    Equal to ``comb(degree + dim, dim)`` =
    ``(d+1)(d+2)(d+3)/6`` for dim=3.
    """
    from math import comb
    return comb(degree + dim, dim)


_UNSET = object()


# ---------------------------------------------------------------------------
# MeshTransfer class
# ---------------------------------------------------------------------------

class MeshTransfer:
    """Sparse regridder between two spherical meshes.

    Builds a sparse weight matrix ``W`` of shape
    ``(receiver.num_nodes, sender.num_nodes)`` so that ``W @ data``
    transfers values from the sender grid to the receiver grid.

    Weights are computed lazily on the first :meth:`transform` call,
    or explicitly via :meth:`build_weights`.

    Args:
        sender: Source mesh (any :class:`Mesh` subclass).
        receiver: Target mesh.
        method: Interpolation method.

            - ``"nearest"``: 1-nearest-neighbor.
            - ``"idw"``: Inverse-distance weighting.
            - ``"gaussian"``: Gaussian distance weighting.
            - ``"barycentric"``: Barycentric on triangulated faces.
            - ``"bilinear"``: Bilinear on quad faces
              (:class:`RectangularMesh` sender only).
            - ``"local_rbf"``: Local RBF interpolation (default).

        k: Number of neighbors for kNN-based methods.
        kernel: RBF kernel name (for ``"local_rbf"``).
        degree: Polynomial augmentation degree (for ``"local_rbf"``).
            0 = constant, 1 = linear, 2 = quadratic, etc.
        max_dist: Maximum neighbor distance. Neighbors beyond this are
            pruned.  Use ``"auto"`` for the 90th-percentile heuristic.

    Example::

        regridder = MeshTransfer(ocean_mesh, target_grid,
                                 method="local_rbf", k=16, degree=0)
        sst_regridded = regridder.transform(sst_ocean)

    Example::

        regridder = MeshTransfer(ocean_mesh, target_grid,
                                 method="local_rbf", k=16, degree=0)
        sst_regridded = regridder.transform(sst_ocean)

        # Or use the @ operator
        result = regridder @ data

        # Access the sparse matrix directly
        W = regridder.weights
    """

    def __init__(
        self,
        sender: Mesh,
        receiver: Mesh,
        method: str = "local_rbf",
        k: int = 16,
        kernel: str = "thin_plate_spline",
        degree: int = 1,
        max_dist: float = None,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.method = method
        self.k = k
        self.kernel = kernel
        self.degree = degree
        self.max_dist = max_dist
        self._weights = None
        self._nearest_senders = None
        self._nearest_distances = None

    def __repr__(self) -> str:
        s = f"MeshTransfer({self.sender.num_nodes:,} \u2192 {self.receiver.num_nodes:,}"
        s += f", method='{self.method}', k={self.k}"
        if self.method == "local_rbf":
            s += f", kernel='{self.kernel}', degree={self.degree}"
        if self.max_dist is not None:
            s += f", max_dist={self.max_dist}"
        built = "built" if self._weights is not None else "not built"
        s += f", {built})"
        return s

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the weight matrix: ``(num_receiver, num_sender)``."""
        return (self.receiver.num_nodes, self.sender.num_nodes)

    @property
    def weights(self) -> sparse.csr_matrix:
        """The sparse weight matrix.  Built lazily on first access."""
        if self._weights is None:
            self.build_weights()
        return self._weights

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def build_weights(
        self,
        method: str = _UNSET,
        k: int = _UNSET,
        kernel: str = _UNSET,
        degree: int = _UNSET,
        max_dist: float = _UNSET,
    ) -> sparse.csr_matrix:
        """Build the sparse interpolation weight matrix.

        When called with no arguments, uses the parameters stored on
        the instance.  Any provided argument updates the stored
        configuration and triggers a rebuild.

        Returns:
            Sparse CSR matrix of shape :attr:`shape`.
        """
        if method is not _UNSET:
            self.method = method
        if k is not _UNSET:
            self.k = k
        if kernel is not _UNSET:
            self.kernel = kernel
        if degree is not _UNSET:
            self.degree = degree
        if max_dist is not _UNSET:
            self.max_dist = max_dist

        n_recv = self.receiver.num_nodes
        n_send = self.sender.num_nodes

        if self.method == "barycentric":
            rows, cols, vals = _barycentric_weights(
                self.sender, self.receiver.nodes
            )
        elif self.method == "bilinear":
            if not isinstance(self.sender, RectangularMesh):
                raise ValueError(
                    "bilinear method requires a RectangularMesh sender"
                )
            rows, cols, vals = _bilinear_weights(
                self.sender, self.receiver.nodes
            )
        else:
            k = self.k
            self._ensure_neighbors(k)
            dists = self._nearest_distances[:, :k]
            idxs = self._nearest_senders[:, :k]

            if self.method in ("nearest", "idw", "gaussian"):
                rows, cols, vals = _knn_weights(dists, idxs, self.method)
            elif self.method == "local_rbf":
                deg = self.degree if self.degree is not None else 1
                max_d = self.max_dist

                if max_d == "auto":
                    mean_d = dists.mean(axis=1)
                    max_d = np.percentile(mean_d, 90)

                if max_d is not None:
                    rows, cols, vals = self._build_weights_thresholded(
                        dists, idxs, self.kernel, deg, max_d,
                    )
                else:
                    rows, cols, vals = _local_rbf_weights(
                        self.sender.nodes,
                        self.receiver.nodes,
                        dists,
                        idxs,
                        kernel=self.kernel,
                        degree=deg,
                    )
            else:
                raise ValueError(f"Unknown method: {self.method}")

        W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_recv, n_send))
        self._weights = W
        return W

    def transform(self, data: NDArray) -> NDArray:
        """Regrid *data* from sender mesh to receiver mesh.

        Builds the weight matrix lazily on first call.

        Args:
            data: Values on the sender grid.  Shape ``(N,)`` for a
                single field, ``(N, d)`` for *d* fields, or
                ``(N, ...)`` for batched data — where ``N`` is
                ``sender.num_nodes``.

        Returns:
            Interpolated values on the receiver grid, same trailing
            dimensions as *data*.
        """
        return self.weights @ data

    def __matmul__(self, data: NDArray) -> NDArray:
        """Allow ``regridder @ data`` as shorthand for :meth:`transform`."""
        return self.transform(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_neighbors(self, k: int, recompute: bool = False):
        """Ensure at least *k* neighbors have been computed."""
        need = (
            self._nearest_senders is None
            or self._nearest_senders.shape[1] < k
            or recompute
        )
        if need:
            self._nearest_distances, self._nearest_senders = query_nearest(
                self.sender.nodes,
                self.receiver.nodes,
                n_neighbors=k,
            )
            if k == 1:
                self._nearest_senders = np.expand_dims(
                    self._nearest_senders, -1
                )
                self._nearest_distances = np.expand_dims(
                    self._nearest_distances, -1
                )

    def _build_weights_thresholded(
        self,
        dists: NDArray,
        idxs: NDArray,
        kernel: str,
        degree: int,
        max_dist: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Build local RBF weights with distance-based neighbor pruning."""
        n_valid = np.sum(dists <= max_dist, axis=1)

        rows_all, cols_all, vals_all = [], [], []
        sender_nodes = self.sender.nodes
        receiver_nodes = self.receiver.nodes

        for nv in np.unique(n_valid):
            if nv == 0:
                continue

            group = np.where(n_valid == nv)[0]

            if nv == 1:
                rows_all.append(group)
                cols_all.append(idxs[group, 0])
                vals_all.append(np.ones(len(group)))
                continue

            deg = degree
            if deg >= 1 and nv < _poly_terms(deg, 3):
                deg = 0

            r, c, v = _local_rbf_weights(
                sender_nodes,
                receiver_nodes[group],
                dists[group, :nv],
                idxs[group, :nv],
                kernel=kernel,
                degree=deg,
            )
            r = group[r]
            rows_all.append(r)
            cols_all.append(c)
            vals_all.append(v)

        if rows_all:
            return (np.concatenate(rows_all),
                    np.concatenate(cols_all),
                    np.concatenate(vals_all))
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
