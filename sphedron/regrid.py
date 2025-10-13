import numpy as np
from scipy.spatial import cKDTree  # type: ignore
from scipy.sparse import csr_matrix
from numpy.linalg import solve
from .mesh.base import Mesh


def latlon_to_unitvec(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])  # (N,3)


def phi_gaussian(r, eps):
    return np.exp(-((r / eps) ** 2))


def phi_multiquadric(r, eps):
    return np.sqrt(1.0 + (r / eps) ** 2)


def phi_inverse_multiquadric(r, eps):
    return 1.0 / np.sqrt(1.0 + (r / eps) ** 2)


def phi_thin_plate(r, _eps_unused):
    # Safeguard r=0 to avoid log(0); limit r->0 equals 0
    out = np.zeros_like(r)
    mask = r > 0
    rr = r[mask]
    out[mask] = (rr**2) * np.log(rr)
    return out


KERNELS = {
    "gaussian": phi_gaussian,
    "mq": phi_multiquadric,
    "imq": phi_inverse_multiquadric,
    "tps": phi_thin_plate,
}


def build_rbf_weights(
    src_lat,
    src_lon,
    dst_lat,
    dst_lon,
    k=64,
    kernel="gaussian",
    eps=None,
    lam=1e-8,  # Tikhonov (units of kernel output)
    degree=0,  # 0 = no polynomial tail; 1 = affine reproduction
    progress=False,
):
    """
    Returns W (Nt x Ns) CSR sparse matrix mapping source->target.
    """
    assert kernel in KERNELS
    phi = KERNELS[kernel]

    Xs = latlon_to_unitvec(src_lat, src_lon)  # (Ns,3)
    Xt = latlon_to_unitvec(dst_lat, dst_lon)  # (Nt,3)
    Ns = Xs.shape[0]
    Nt = Xt.shape[0]

    # kNN on the unit sphere (Euclidean = chord distance)
    tree = cKDTree(Xs)
    dists, idxs = tree.query(Xt, k=k, workers=-1)  # dists: (Nt,k), idxs: (Nt,k)

    # Choose eps if not given: median neighbor spacing
    if eps is None:
        # Using median of nonzero distances for stability
        # skip the first neighbor if source==target
        base = np.median(dists[:, 1:])
        eps = max(base, 1e-6)

    # Pre-allocate sparse structure
    indptr = np.zeros(Nt + 1, dtype=np.int64)
    indices = np.zeros(Nt * k, dtype=np.int64)
    data = np.zeros(Nt * k, dtype=np.float64)

    # Build rows independently
    offset = 0
    for i in range(Nt):
        nbr_idx = idxs[i]  # (k,)
        nbr_pts = Xs[nbr_idx]  # (k,3)
        rmat = np.linalg.norm(
            nbr_pts[None, :, :] - nbr_pts[:, None, :], axis=-1
        )  # (k,k)
        K = phi(rmat, eps)
        if lam > 0:
            K.flat[:: k + 1] += lam  # add lam to diagonal

        rt = np.linalg.norm(nbr_pts - Xt[i], axis=1)  # (k,)
        phit = phi(rt, eps)  # (k,)

        if degree == 0:
            # Solve K w = phit
            w = solve(K, phit)
        else:
            # Augment with affine polynomial [1, x, y, z]
            P = np.column_stack([np.ones(k), nbr_pts])  # (k, 4)
            A = np.block([[K, P], [P.T, np.zeros((4, 4))]])  # (k+4, k+4)
            rhs = np.concatenate([phit, np.array([1.0, *Xt[i]])])  # (k+4,)
            sol = solve(A, rhs)
            w = sol[:k]  # discard Lagrange multipliers

        # Fill sparse row
        nnz = k
        indices[offset : offset + nnz] = nbr_idx
        data[offset : offset + nnz] = w
        indptr[i + 1] = indptr[i] + nnz
        offset += nnz
        if progress and (i % 1000 == 0):
            print(f"{i}/{Nt} targets")

    W = csr_matrix((data, indices, indptr), shape=(Nt, Ns))
    meta = dict(
        kernel=kernel,
        eps=float(eps),
        k=int(k),
        lam=float(lam),
        degree=int(degree),
    )
    return W, meta


def build_rbf_weights_from_meshes(
    src_mesh: Mesh,
    dst_mesh: Mesh,
    k: int = 64,
    kernel: str = "gaussian",
    eps: float | None = None,
    lam: float = 1e-8,
    degree: int = 0,
    progress: bool = False,
):
    """
    Convenience wrapper to build RBF weights using Mesh instances.
    """
    src_ll = src_mesh.nodes_latlong  # degrees, shape (Ns, 2)
    dst_ll = dst_mesh.nodes_latlong  # degrees, shape (Nt, 2)
    Ns = src_ll.shape[0]
    k = int(min(max(1, k), Ns))  # clamp to [1, Ns]
    W, meta = build_rbf_weights(
        src_lat=src_ll[:, 0],
        src_lon=src_ll[:, 1],
        dst_lat=dst_ll[:, 0],
        dst_lon=dst_ll[:, 1],
        k=k,
        kernel=kernel,
        eps=eps,
        lam=lam,
        degree=degree,
        progress=progress,
    )
    meta.update(Ns=int(Ns), Nt=int(dst_ll.shape[0]))
    return W, meta


class RBFRegridder:
    """
    Regrid values between meshes using RBF weights built once.

    - Supports values shaped:
      (Ns,), (Ns, d) -> returns (Nt,), (Nt, d)
      (T, Ns)        -> returns (T, Nt)
    """

    def __init__(
        self,
        src_mesh: Mesh,
        dst_mesh: Mesh,
        k: int = 64,
        kernel: str = "gaussian",
        eps: float | None = None,
        lam: float = 1e-8,
        degree: int = 0,
        progress: bool = False,
    ):
        self.src_mesh = src_mesh
        self.dst_mesh = dst_mesh
        self.W, self.meta = build_rbf_weights_from_meshes(
            src_mesh,
            dst_mesh,
            k=k,
            kernel=kernel,
            eps=eps,
            lam=lam,
            degree=degree,
            progress=progress,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        Apply the precomputed weights to sender values.
        """
        Ns = self.meta["Ns"]
        if values.ndim == 1:
            if values.shape[0] != Ns:
                raise ValueError(
                    "Expected shape (Ns,), got %r" % (values.shape,)
                )
            return self.W @ values
        if values.ndim == 2:
            # (Ns, d) or (T, Ns)
            if values.shape[0] == Ns:
                return self.W @ values  # -> (Nt, d)
            if values.shape[1] == Ns:
                return values @ self.W.T  # -> (T, Nt)
        raise ValueError(
            "Unsupported shape %r. Use (Ns,), (Ns, d) or (T, Ns)."
            % (values.shape,)
        )


# Example usage:
# W, meta = build_rbf_weights(src_lat, src_lon, dst_lat, dst_lon, k=64, kernel="gaussian", degree=0)
# save_npz("rbf_weights.npz", W)
# Later:
# W = load_npz("rbf_weights.npz")
# # Apply to one time slice flattened to (Ns,)
# dst_vals = W @ src_vals
# # Or to (T, Ns)
# dst_all = (W @ src_all.T).T  # -> (T, Nt)
# Example (in your notebook/script)
# from sphedron.mesh import Icosphere, UniformMesh
# from sphedron.regrid import RBFRegridder

# src = Icosphere.from_base(refine_factor=8)   # sender mesh
# dst = UniformMesh(resolution=1.0)            # receiver mesh (1-degree)

# regridder = RBFRegridder(src, dst, k=64, kernel="gaussian", degree=0)

# # Single field (Ns,) -> (Nt,)
# vals_src = np.random.rand(src.num_nodes)
# vals_dst = regridder.transform(vals_src)

# # Multichannel (Ns, d) -> (Nt, d)
# vals_src_2d = np.random.rand(src.num_nodes, 3)
# vals_dst_2d = regridder.transform(vals_src_2d)

# # Time x nodes (T, Ns) -> (T, Nt)
# vals_time = np.random.rand(12, src.num_nodes)
# vals_time_dst = regridder.transform(vals_time)
