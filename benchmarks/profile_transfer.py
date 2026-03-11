"""Profile and validate sparse mesh transfer: Icosphere → UniformMesh."""

import cProfile
import pstats
import time
import numpy as np
from sphedron import Icosphere, UniformMesh
from sphedron.transfer import MeshTransfer
from sphedron.transform import xyz_to_thetaphi


def spherical_field(nodes):
    """Compute a test field: f(theta, phi) = cos(3*theta) * sin(2*phi)."""
    tp = xyz_to_thetaphi(nodes)
    return np.cos(3 * tp[:, 0]) * np.sin(2 * tp[:, 1])


def validate():
    """Compare all sparse weight methods on Icosphere→UniformMesh transfer."""
    print("Building meshes...")
    t0 = time.perf_counter()
    ico = Icosphere.from_base(refine_factor=64)
    dt_ico = time.perf_counter() - t0
    print("  Icosphere factor=64: {:,} nodes ({:.2f}s)".format(
        ico.num_nodes, dt_ico))

    t0 = time.perf_counter()
    uni = UniformMesh(resolution=0.5)
    dt_uni = time.perf_counter() - t0
    print("  UniformMesh 0.5deg:  {:,} nodes ({:.2f}s)".format(
        uni.num_nodes, dt_uni))

    x_send = spherical_field(ico.nodes)
    x_true = spherical_field(uni.nodes)
    print("  Field: min={:.4f}, max={:.4f}\n".format(
        x_send.min(), x_send.max()))

    transfer = MeshTransfer(ico, uni, n_neighbors=16)

    configs = [
        ("nearest",             dict(method="nearest")),
        ("idw(k=5)",            dict(method="idw", k=5)),
        ("gaussian(k=5)",      dict(method="gaussian", k=5)),
        ("barycentric",         dict(method="barycentric")),
        ("local_rbf(k=8,d=0)", dict(method="local_rbf", k=8, degree=0)),
        ("local_rbf(k=8,d=1)", dict(method="local_rbf", k=8, degree=1)),
        ("local_rbf(k=16,d=1)", dict(method="local_rbf", k=16, degree=1)),
    ]

    header = "{:<25s} {:>12s} {:>8s} {:>8s} {:>8s}".format(
        "method", "RMSE", "nnz/row", "build", "apply")
    print(header)
    print("-" * len(header))

    for name, kwargs in configs:
        t0 = time.perf_counter()
        W = transfer.build_weights(**kwargs)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        y = W @ x_send
        t_apply = time.perf_counter() - t0

        rmse = np.sqrt(np.mean((y - x_true) ** 2))
        nnz = W.nnz / W.shape[0]
        print("{:<25s} {:>12.6e} {:>8.1f} {:>7.3f}s {:>7.4f}s".format(
            name, rmse, nnz, t_build, t_apply))

    # Reference: scipy RBFInterpolator (dense, for comparison)
    print()
    for k in [8, 16]:
        t0 = time.perf_counter()
        from scipy.interpolate import RBFInterpolator
        y_rbf = RBFInterpolator(
            ico.nodes, x_send,
            kernel="thin_plate_spline", neighbors=k,
        )(uni.nodes)
        dt = time.perf_counter() - t0
        rmse = np.sqrt(np.mean((y_rbf - x_true) ** 2))
        print("scipy_RBF(k={:<2d})            {:>12.6e}              {:>7.3f}s".format(
            k, rmse, dt))


def profile_local_rbf():
    """Detailed cProfile of local_rbf weight building."""
    ico = Icosphere.from_base(refine_factor=64)
    uni = UniformMesh(resolution=0.5)

    transfer = MeshTransfer(ico, uni, n_neighbors=8)

    profiler = cProfile.Profile()
    profiler.enable()
    W = transfer.build_weights(method="local_rbf", k=8, degree=1)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("tottime")
    print("\n=== cProfile: build_weights(local_rbf, k=8, d=1) ===")
    stats.print_stats(15)

    # Profile apply
    x_send = spherical_field(ico.nodes)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):
        y = W @ x_send
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("tottime")
    print("\n=== cProfile: W @ x (10 iterations) ===")
    stats.print_stats(10)


if __name__ == "__main__":
    print("=== Sparse Weight Methods Comparison ===\n")
    validate()
    profile_local_rbf()
