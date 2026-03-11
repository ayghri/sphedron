Mesh Transfer
=============

Sphedron provides tools to transfer scalar fields between meshes of
different resolutions and topologies.  The core idea is to precompute a
**sparse weight matrix** ``W`` of shape ``(n_receiver, n_sender)`` so
that transferring a field is a single sparse matrix–vector product:

.. math::

   \mathbf{y} = W \, \mathbf{x}

Each row of ``W`` has a fixed number of non-zero entries (the *degree*),
making it suitable as a differentiable linear layer in deep-learning
pipelines.


Quick start
-----------

.. code-block:: python

   from sphedron import Icosphere, UniformMesh
   from sphedron.transfer import MeshTransfer

   sender   = Icosphere.from_base(refine_factor=256)   # 655,362 nodes
   receiver = UniformMesh(resolution=0.5)               # 259,200 nodes

   # Create the regridder — all config in the constructor
   regridder = MeshTransfer(sender, receiver,
                            method="local_rbf", k=16, degree=0)

   # Transfer a field (weights are built lazily on first call)
   y = regridder.transform(x_sender)

   # Equivalent shorthand via @ operator
   y = regridder @ x_sender

   # Access the sparse matrix directly
   W = regridder.weights


The ``MeshTransfer`` object
---------------------------

``MeshTransfer`` is the main entry point for regridding.  It accepts
the source and target meshes together with all interpolation parameters:

.. code-block:: python

   regridder = MeshTransfer(
       sender,                       # any Mesh subclass or NodesOnlyMesh
       receiver,                     # target mesh
       method="local_rbf",           # interpolation method
       k=16,                         # number of neighbors
       kernel="thin_plate_spline",   # RBF kernel
       degree=0,                     # polynomial augmentation degree
   )

Key properties and methods:

- ``regridder.transform(data)`` — regrid data (lazy weight build).
- ``regridder @ data`` — same as ``transform``.
- ``regridder.weights`` — the sparse CSR matrix (lazy build).
- ``regridder.shape`` — ``(n_receiver, n_sender)``.
- ``regridder.build_weights(...)`` — explicit build; any argument
  updates the stored configuration.
- ``repr(regridder)`` — shows config and build status::

    MeshTransfer(86,096 → 64,800, method='local_rbf', k=16,
                 kernel='thin_plate_spline', degree=0, built)


Interpolation methods
---------------------

The ``method`` parameter supports several strategies:

.. list-table::
   :header-rows: 1
   :widths: 18 8 74

   * - Method
     - Degree
     - Description
   * - ``nearest``
     - 1
     - Nearest-neighbor assignment. Fast but blocky.
   * - ``idw``
     - *k*
     - Inverse distance weighting: :math:`w_i = 1/d_i`, row-normalized.
   * - ``gaussian``
     - *k*
     - Gaussian kernel: :math:`w_i = \exp(-0.2\,(d_i/d_\min)^2)`,
       row-normalized.
   * - ``barycentric``
     - 3
     - Projects onto the nearest triangle and computes barycentric
       coordinates.  Exact for linear fields on the mesh surface but
       requires a ``trimesh`` proximity query (slower to build).
   * - ``bilinear``
     - 4
     - Bilinear interpolation on quad faces (``RectangularMesh`` only).
   * - ``local_rbf``
     - *k*
     - Local radial basis function interpolation with polynomial
       augmentation (default).  Best accuracy.


Local RBF interpolation
-----------------------

The ``local_rbf`` method builds, for each receiver node, a small k×k
RBF kernel system from its *k* nearest sender neighbors:

.. math::

   \begin{bmatrix} \Phi & P \\ P^T & 0 \end{bmatrix}
   \begin{bmatrix} \mathbf{w} \\ \mathbf{c} \end{bmatrix}
   =
   \begin{bmatrix} \boldsymbol{\phi} \\ \mathbf{p}_q \end{bmatrix}

where :math:`\Phi_{ij} = \varphi(\|x_i - x_j\|)` is the kernel matrix
among the *k* neighbors, :math:`\boldsymbol{\phi}_i = \varphi(\|x_i - q\|)`
evaluates the kernel at the query point *q*, and *P* contains polynomial
terms ensuring:

- **degree=0**: partition of unity (weights sum to 1, constant fields
  reproduced exactly)
- **degree=1**: linear reproduction (linear fields reproduced exactly)
- **degree=2+**: higher-order polynomial reproduction

All *M* independent systems are solved in a single batched
``np.linalg.solve`` call, making this significantly faster than
scipy's ``RBFInterpolator`` while producing identical results.

Available kernels (all conditionally positive definite):

- ``thin_plate_spline`` (default): :math:`\varphi(r) = r^2 \ln r`
- ``cubic``: :math:`\varphi(r) = r^3`
- ``linear``: :math:`\varphi(r) = r`

For degree ≥ 1, a safety mechanism detects ill-conditioned local systems
(e.g. near coastlines with degenerate neighbor configurations) and falls
back to degree=0.


Benchmark: spherical harmonics
------------------------------

Transfer of a spherical harmonic test field
:math:`f(\theta, \varphi) = \cos(3\theta)\sin(2\varphi)` from an
Icosphere (655,362 nodes, factor=256) to a UniformMesh (259,200 nodes,
0.5° resolution):

.. list-table::
   :header-rows: 1
   :widths: 25 12 14 10 10 10

   * - Method
     - RMSE
     - ||delta||/||y||
     - nnz/row
     - Build
     - Apply
   * - ``nearest``
     - 3.64e-2
     - 7.3%
     - 1
     - 0.20s
     - 1.0ms
   * - ``idw(k=5)``
     - 3.45e-2
     - 6.9%
     - 5
     - 0.04s
     - 2.1ms
   * - ``gaussian(k=5)``
     - 3.15e-2
     - 6.3%
     - 5
     - 0.05s
     - 2.1ms
   * - ``barycentric``
     - 2.20e-2
     - 4.4%
     - 3
     - 81.3s
     - 1.3ms
   * - ``local_rbf(k=16, d=0)``
     - 2.20e-2
     - 4.4%
     - 16
     - 2.6s
     - 3.1ms

``local_rbf(k=16, d=0)`` achieves the same accuracy as ``barycentric``
(4.4% relative error) in **2.6s vs 81s** — a 31× speedup on weight
construction, with apply times under 3ms.


Benchmark: CESM2 real-data roundtrip
-------------------------------------

A more rigorous test: regrid 5 CESM2 ocean variables (SSH, SST, SHF,
TAUX, TAUY) from the curvilinear POP grid (86,096 ocean cells) through
an intermediate grid and back.  We subtract the temporal mean (1332
monthly snapshots, 1990–2100) and measure the **centered-field roundtrip
error** — the relative error on anomalies, averaged over all timesteps.

.. list-table:: Uniform 1° intermediate (64,800 nodes)
   :header-rows: 1
   :widths: 26 10 10 10 10 10 10 10 12

   * - Config
     - SSH
     - SST
     - SHF
     - TAUX
     - TAUY
     - Mean
     - Build
     - Apply
   * - xESMF bilinear
     - 7.78%
     - 1.50%
     - 3.93%
     - 2.60%
     - 2.70%
     - 3.70%
     - —
     - 38ms
   * - ``nearest``
     - 12.81%
     - 4.00%
     - 6.74%
     - 7.45%
     - 6.82%
     - 7.56%
     - 0.0s
     - 0.21ms
   * - ``rbf tps k=8 d=0``
     - 7.05%
     - 1.06%
     - 3.12%
     - 1.79%
     - 1.92%
     - 2.99%
     - 0.5s
     - 0.34ms
   * - ``rbf tps k=16 d=0``
     - 6.65%
     - 0.84%
     - 2.79%
     - 1.22%
     - 1.42%
     - 2.58%
     - 1.5s
     - 0.47ms
   * - ``rbf tps k=32 d=0``
     - 6.57%
     - 0.75%
     - 2.69%
     - 1.00%
     - 1.22%
     - 2.44%
     - 5.3s
     - 0.74ms
   * - ``rbf cubic k=16 d=0``
     - 6.89%
     - 0.86%
     - 2.87%
     - 1.31%
     - 1.51%
     - 2.69%
     - 1.4s
     - 0.47ms

.. list-table:: Icosphere-128 intermediate (163,842 nodes)
   :header-rows: 1
   :widths: 26 10 10 10 10 10 10 10 12

   * - Config
     - SSH
     - SST
     - SHF
     - TAUX
     - TAUY
     - Mean
     - Build
     - Apply
   * - xESMF bilinear
     - 7.78%
     - 1.50%
     - 3.93%
     - 2.60%
     - 2.70%
     - 3.70%
     - —
     - 38ms
   * - ``nearest``
     - 7.53%
     - 1.99%
     - 3.29%
     - 3.10%
     - 3.59%
     - 3.90%
     - 0.1s
     - 0.36ms
   * - ``rbf tps k=8 d=0``
     - 5.81%
     - 0.35%
     - 1.37%
     - 0.55%
     - 0.73%
     - 1.76%
     - 0.8s
     - 0.56ms
   * - ``rbf tps k=16 d=0``
     - 5.77%
     - 0.27%
     - 1.34%
     - 0.40%
     - 0.57%
     - 1.67%
     - 2.6s
     - 0.77ms
   * - ``rbf tps k=32 d=0``
     - 5.80%
     - 0.23%
     - 1.27%
     - 0.33%
     - 0.53%
     - 1.63%
     - 8.9s
     - 1.19ms
   * - ``rbf cubic k=16 d=0``
     - 5.84%
     - 0.25%
     - 1.27%
     - 0.39%
     - 0.55%
     - 1.66%
     - 2.4s
     - 0.77ms

Key observations:

- **Sphedron beats xESMF bilinear on every variable** — 2.44% vs 3.70%
  mean at Uniform 1°, and **1.67%** at Icosphere-128.
- **Apply is 14–50× faster** than xESMF's official ``regridder(data)``
  API (sparse matmul vs ESMF's internal routines).
- **TPS kernel with degree=0** is the best overall choice.
- SSH has the highest error (~5.8% at Icosphere-128) due to fine-scale
  ocean dynamics; all other variables are under 1.4%.
- Increasing *k* beyond 16 gives diminishing returns — the error floor
  is set by the intermediate grid resolution, not interpolation order.


Use in deep learning
--------------------

Since the weight matrix ``W`` is a fixed sparse matrix, the transfer
operation :math:`y = Wx` can be plugged directly into a deep-learning
graph as a sparse linear layer with no learnable parameters:

.. code-block:: python

   import torch

   # Convert to PyTorch sparse tensor
   W_coo = regridder.weights.tocoo()
   indices = torch.LongTensor(np.vstack([W_coo.row, W_coo.col]))
   values  = torch.FloatTensor(W_coo.data)
   W_torch = torch.sparse_coo_tensor(indices, values, W_coo.shape)

   # Differentiable transfer
   y = torch.sparse.mm(W_torch, x)   # gradient flows through x
