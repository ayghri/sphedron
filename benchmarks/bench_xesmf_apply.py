"""Measure xESMF official-API apply times vs Sphedron sparse matmul."""
import sys
import time
import numpy as np
import xarray as xr
import xesmf as xe
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from sphedron import UniformMesh
from sphedron.mesh.base import NodesOnlyMesh
from sphedron.transfer import MeshTransfer


def main():
    cesm2_path = Path("/buckets/datasets/ssh/simulations/cesm2/monthly/merged")
    ds = xr.open_dataset(cesm2_path / "SST" / "1281.014.nc").isel(time=0)
    tlat = ds["TLAT"].values
    tlong = ds["TLONG"].values
    sst = ds["SST"].values
    ocean_mask = ~np.isnan(sst)
    print(f"Data loaded: {sst.shape}, {ocean_mask.sum()} ocean pts", flush=True)

    # xESMF grids
    ds_in = xr.Dataset(
        {"lat": (["nlat", "nlon"], tlat), "lon": (["nlat", "nlon"], tlong)}
    )
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.5, 90, 1.0)),
            "lon": (["lon"], np.arange(0.5, 360, 1.0)),
        }
    )
    ds_in_bwd = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.5, 90, 1.0)),
            "lon": (["lon"], np.arange(0.5, 360, 1.0)),
        }
    )
    ds_out_bwd = xr.Dataset(
        {"lat": (["nlat", "nlon"], tlat), "lon": (["nlat", "nlon"], tlong)}
    )

    xe_fwd_bl = xe.Regridder(ds_in, ds_out, "bilinear", periodic=True)
    xe_fwd_nn = xe.Regridder(ds_in, ds_out, "nearest_s2d", periodic=True)
    xe_bwd_bl = xe.Regridder(ds_in_bwd, ds_out_bwd, "bilinear", periodic=True)
    print("Regridders built", flush=True)

    # Warmup
    _ = xe_fwd_bl(sst)
    _ = xe_fwd_nn(sst)
    intermediate = xe_fwd_bl(sst)
    _ = xe_bwd_bl(intermediate)

    n_iter = 20

    def median_time(func, n=n_iter):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            func()
            times.append(time.perf_counter() - t0)
        return np.median(times)

    fwd_bl = median_time(lambda: xe_fwd_bl(sst))
    print(f"xESMF fwd bilinear:  {fwd_bl*1e3:.2f}ms", flush=True)

    fwd_nn = median_time(lambda: xe_fwd_nn(sst))
    print(f"xESMF fwd nearest:   {fwd_nn*1e3:.2f}ms", flush=True)

    bwd_bl = median_time(lambda: xe_bwd_bl(intermediate))
    print(f"xESMF bwd bilinear:  {bwd_bl*1e3:.2f}ms", flush=True)

    rt_bl = median_time(lambda: xe_bwd_bl(xe_fwd_bl(sst)))
    print(f"xESMF roundtrip:     {rt_bl*1e3:.2f}ms", flush=True)

    # Sphedron
    latlon_ocean = np.column_stack([tlat[ocean_mask], tlong[ocean_mask]])
    sender = NodesOnlyMesh(latlon_ocean)
    receiver = UniformMesh(resolution=1.0)
    sst_ocean = sst[ocean_mask]
    transfer_fwd = MeshTransfer(sender, receiver)
    transfer_bwd = MeshTransfer(receiver, sender)

    print("\nSphedron roundtrip (single field):", flush=True)
    for k in [8, 16, 32]:
        W_f = transfer_fwd.build_weights(method="local_rbf", k=k, degree=0)
        W_b = transfer_bwd.build_weights(method="local_rbf", k=k, degree=0)
        _ = W_b @ (W_f @ sst_ocean)
        rt_sph = median_time(lambda: W_b @ (W_f @ sst_ocean))
        print(
            f"  k={k:>2d}: {rt_sph*1e3:.2f}ms  ({rt_bl/rt_sph:.0f}x faster)",
            flush=True,
        )


if __name__ == "__main__":
    main()
