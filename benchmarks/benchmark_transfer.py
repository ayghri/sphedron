"""Benchmark sparse interpolation methods for mesh-to-mesh transfer.

Compares all available ``build_weights`` methods on an
Icosphere -> UniformMesh transfer using a spherical harmonic test field.

Usage::

    python benchmarks/benchmark_transfer.py
    python benchmarks/benchmark_transfer.py --factor 128
"""

import argparse
import time
import numpy as np
from sphedron import Icosphere, UniformMesh
from sphedron.transfer import MeshTransfer
from sphedron.transform import xyz_to_thetaphi


def spherical_field(nodes):
    """Test field: f(theta, phi) = cos(3*theta) * sin(2*phi)."""
    tp = xyz_to_thetaphi(nodes)
    return np.cos(3 * tp[:, 0]) * np.sin(2 * tp[:, 1])


def run_benchmark(factor=256, resolution=0.5):
    """Run the full benchmark and print results."""
    print("Building meshes...")
    t0 = time.perf_counter()
    ico = Icosphere.from_base(refine_factor=factor)
    dt_ico = time.perf_counter() - t0

    t0 = time.perf_counter()
    uni = UniformMesh(resolution=resolution)
    dt_uni = time.perf_counter() - t0

    print("  Icosphere factor={}: {:,} nodes ({:.2f}s)".format(
        factor, ico.num_nodes, dt_ico))
    print("  UniformMesh {}deg:  {:,} nodes ({:.2f}s)".format(
        resolution, uni.num_nodes, dt_uni))

    # Compute test field on both meshes
    x_send = spherical_field(ico.nodes)
    x_true = spherical_field(uni.nodes)
    norm_true = np.linalg.norm(x_true)
    print("  Field range: [{:.4f}, {:.4f}]\n".format(
        x_send.min(), x_send.max()))

    transfer = MeshTransfer(ico, uni, n_neighbors=16)

    configs = [
        ("nearest",             dict(method="nearest")),
        ("idw(k=5)",            dict(method="idw", k=5)),
        ("gaussian(k=5)",       dict(method="gaussian", k=5)),
        ("local_rbf(k=8,d=0)",  dict(method="local_rbf", k=8, degree=0)),
        ("local_rbf(k=8,d=1)",  dict(method="local_rbf", k=8, degree=1)),
        ("local_rbf(k=16,d=0)", dict(method="local_rbf", k=16, degree=0)),
        ("local_rbf(k=16,d=1)", dict(method="local_rbf", k=16, degree=1)),
    ]

    header = "{:<25s} {:>12s} {:>12s} {:>8s} {:>8s} {:>8s}".format(
        "method", "RMSE", "||d||/||y||", "nnz/row", "build", "apply")
    print(header)
    print("-" * len(header))

    for name, kwargs in configs:
        t0 = time.perf_counter()
        W = transfer.build_weights(**kwargs)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        y = W @ x_send
        t_apply = time.perf_counter() - t0

        delta = y - x_true
        rmse = np.sqrt(np.mean(delta ** 2))
        rel_norm = np.linalg.norm(delta) / norm_true
        nnz = W.nnz / W.shape[0]
        print("{:<25s} {:>12.6e} {:>12.6e} {:>8.1f} {:>7.3f}s {:>7.4f}s".format(
            name, rmse, rel_norm, nnz, t_build, t_apply))

    # Reference: scipy RBFInterpolator
    print()
    from scipy.interpolate import RBFInterpolator
    for k in [8, 16]:
        t0 = time.perf_counter()
        y_rbf = RBFInterpolator(
            ico.nodes, x_send,
            kernel="thin_plate_spline", neighbors=k,
        )(uni.nodes)
        dt = time.perf_counter() - t0
        delta = y_rbf - x_true
        rmse = np.sqrt(np.mean(delta ** 2))
        rel_norm = np.linalg.norm(delta) / norm_true
        print("{:<25s} {:>12.6e} {:>12.6e} {:>8s} {:>7.3f}s".format(
            "scipy_RBF(k={})".format(k), rmse, rel_norm, "", dt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark sparse mesh transfer methods")
    parser.add_argument(
        "--factor", type=int, default=256,
        help="Icosphere refinement factor (default: 256)")
    parser.add_argument(
        "--resolution", type=float, default=0.5,
        help="UniformMesh resolution in degrees (default: 0.5)")
    args = parser.parse_args()

    print("=== Sparse Transfer Methods Benchmark ===\n")
    run_benchmark(factor=args.factor, resolution=args.resolution)
