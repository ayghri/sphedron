"""Profile and validate Icosphere refinement: vectorized vs reference."""

import cProfile
import pstats
import time
import numpy as np
from sphedron import Icosphere
from sphedron.refine import refine_triangles, _refine_triangles_reference

FACTORS = [64, 128, 256, 512]


def validate(factor=64):
    """Check vectorized matches reference for both use_angle modes."""
    nodes, faces = Icosphere.base()
    from sphedron.transform import rotate_nodes
    nodes = rotate_nodes(nodes, axis="y", angles=Icosphere.rotation_angle)

    for use_angle in [False, True]:
        n_vec, f_vec = refine_triangles(nodes, faces, factor, use_angle)
        n_ref, f_ref = _refine_triangles_reference(nodes, faces, factor, use_angle)

        node_err = np.max(np.abs(n_vec - n_ref))
        faces_match = np.array_equal(f_vec, f_ref)
        label = f"use_angle={use_angle}"
        print(f"  {label}: nodes max_err={node_err:.2e}, "
              f"faces_match={faces_match}")
        assert node_err < 1e-12, f"{label}: node mismatch {node_err}"
        assert faces_match, f"{label}: face mismatch"


def benchmark():
    """Wall-clock comparison: vectorized vs reference."""
    nodes, faces = Icosphere.base()
    from sphedron.transform import rotate_nodes
    nodes = rotate_nodes(nodes, axis="y", angles=Icosphere.rotation_angle)

    print(f"\n{'factor':>6}  {'nodes':>10}  {'vectorized':>10}  "
          f"{'reference':>10}  {'speedup':>8}")
    print("-" * 56)

    for factor in FACTORS:
        # Vectorized
        t0 = time.perf_counter()
        n_vec, f_vec = refine_triangles(nodes, faces, factor)
        dt_vec = time.perf_counter() - t0

        # Reference
        t0 = time.perf_counter()
        n_ref, f_ref = _refine_triangles_reference(nodes, faces, factor)
        dt_ref = time.perf_counter() - t0

        speedup = dt_ref / dt_vec if dt_vec > 0 else float("inf")
        print(f"{factor:>6}  {n_vec.shape[0]:>10,}  {dt_vec:>9.3f}s  "
              f"{dt_ref:>9.3f}s  {speedup:>7.1f}x")


def profile_vectorized(factor=512):
    """Detailed cProfile of the vectorized version."""
    nodes, faces = Icosphere.base()
    from sphedron.transform import rotate_nodes
    nodes = rotate_nodes(nodes, axis="y", angles=Icosphere.rotation_angle)

    profiler = cProfile.Profile()
    profiler.enable()
    refine_triangles(nodes, faces, factor)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("tottime")
    print(f"\n=== cProfile vectorized, factor={factor} ===")
    stats.print_stats(20)


if __name__ == "__main__":
    print("=== Validation (factor=64) ===")
    validate(64)
    print("  PASSED\n")

    print("=== Benchmark ===")
    benchmark()

    profile_vectorized(512)
