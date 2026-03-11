"""
Simple benchmark script comparing xESMF CPU vs GPU regridding.
"""

import time
import numpy as np
import xarray as xr
import xesmf as xe

import cupy as cp
import cupyx.scipy.sparse as cp_sparse




def create_test_data():
    """Create simple test grids and data."""
    # Input grid: 1 degree resolution (360x180)
    ds_in = xr.Dataset(
        {
            "lon": (["lon"], np.linspace(-179.5, 179.5, 360)),
            "lat": (["lat"], np.linspace(-89.5, 89.5, 180)),
        }
    )

    # Output grid: 0.5 degree resolution (720x360)
    ds_out = xr.Dataset(
        {
            "lon": (["lon"], np.linspace(-179.75, 179.75, 720)),
            "lat": (["lat"], np.linspace(-89.75, 89.75, 360)),
        }
    )

    # Create sample data: 10 time steps
    lon_2d, lat_2d = np.meshgrid(ds_in.lon.values, ds_in.lat.values)
    data = np.zeros((10, 180, 360))
    for t in range(10):
        data[t] = np.sin(lon_2d * np.pi / 180) * np.cos(lat_2d * np.pi / 180)

    return ds_in, ds_out, data


def benchmark_cpu(regridder, data, n_iter=10):
    """Benchmark CPU regridding."""
    times = []

    # Warmup
    _ = regridder(data)

    # Benchmark
    for _ in range(n_iter):
        start = time.perf_counter()
        _ = regridder(data)
        times.append(time.perf_counter() - start)

    return np.array(times)


def benchmark_gpu(regridder, data, n_iter=10):
    """Benchmark GPU regridding."""

    # Prepare weights on GPU
    weights_coo = regridder.weights.data
    weights_gpu = cp_sparse.csr_matrix(
        (
            cp.asarray(weights_coo.data),
            (
                cp.asarray(weights_coo.coords[0]),
                cp.asarray(weights_coo.coords[1]),
            ),
        ),
        shape=weights_coo.shape,
    )

    shape_out = regridder.shape_out

    times = []

    # Warmup
    data_gpu = cp.asarray(data)
    extra_shape = data_gpu.shape[:-2]
    indata_flat = data_gpu.reshape(*extra_shape, -1)
    n_extra = int(np.prod(extra_shape))
    indata_2d = indata_flat.reshape(n_extra, -1).T
    _ = weights_gpu @ indata_2d
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    for _ in range(n_iter):
        start = time.perf_counter()

        # Transfer to GPU
        data_gpu = cp.asarray(data)

        # Reshape and compute
        indata_flat = data_gpu.reshape(*extra_shape, -1)
        indata_2d = indata_flat.reshape(n_extra, -1).T
        outdata_2d = weights_gpu @ indata_2d
        _ = outdata_2d.T.reshape(*extra_shape, shape_out[0], shape_out[1])

        # Transfer back

        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - start)

    return np.array(times)


def main():
    print("\n" + "=" * 60)
    print("Simple xESMF CPU vs GPU Benchmark")
    print("=" * 60)

    # Create test data
    print("\n1. Creating test grids and data...")
    ds_in, ds_out, data = create_test_data()
    print(
        f"   Input grid:  {len(ds_in.lon)}x{len(ds_in.lat)} = {len(ds_in.lon) * len(ds_in.lat):,} cells"
    )
    print(
        f"   Output grid: {len(ds_out.lon)}x{len(ds_out.lat)} = {len(ds_out.lon) * len(ds_out.lat):,} cells"
    )
    print(f"   Data shape:  {data.shape} ({data.nbytes / 1024 / 1024:.1f} MB)")

    # Create regridder
    print("\n2. Computing regridding weights (CPU, ESMF)...")
    start = time.perf_counter()
    regridder = xe.Regridder(ds_in, ds_out, "bilinear", periodic=False)
    weight_time = time.perf_counter() - start
    print(f"   Done in {weight_time:.3f}s")
    print(f"   Weight matrix: {regridder.weights.data.nnz:,} non-zero entries")

    # Benchmark CPU
    print("\n3. Benchmarking CPU regridding...")
    cpu_times = benchmark_cpu(regridder, data, n_iter=10)
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    print(f"   Time: {cpu_mean * 1000:.2f} ± {cpu_std * 1000:.2f} ms")

    # Benchmark GPU
    print("\n4. Benchmarking GPU regridding...")
    gpu_times = benchmark_gpu(regridder, data, n_iter=10)
    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    print(f"   Time: {gpu_mean * 1000:.2f} ± {gpu_std * 1000:.2f} ms")

    speedup = cpu_mean / gpu_mean
    print(f"\n{'=' * 60}")
    print(f"GPU Speedup: {speedup:.2f}x")
    print(f"{'=' * 60}")

    if speedup > 1:
        print(f"✓ GPU is {speedup:.2f}x faster!")
    else:
        print("✗ GPU is slower (data too small for GPU benefit)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
