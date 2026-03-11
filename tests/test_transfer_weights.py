import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import RBFInterpolator
from sphedron import Icosphere, UniformMesh
from sphedron.transfer import MeshTransfer
from sphedron.transform import xyz_to_thetaphi


@pytest.fixture(scope="module")
def meshes():
    """Create sender (Icosphere) and receiver (UniformMesh) meshes."""
    sender = Icosphere.from_base(refine_factor=6)
    receiver = UniformMesh(resolution=5.0)
    return sender, receiver


@pytest.fixture(scope="module")
def regridder(meshes):
    """MeshTransfer instance with default local_rbf config."""
    sender, receiver = meshes
    return MeshTransfer(sender, receiver, k=16)


class TestSparseWeightShape:
    """Verify sparse matrix shape and sparsity pattern."""

    @pytest.mark.parametrize("method,expected_k", [
        ("nearest", 1),
        ("idw", 5),
        ("gaussian", 5),
        ("local_rbf", 8),
    ])
    def test_shape_and_nnz(self, regridder, meshes, method, expected_k):
        sender, receiver = meshes
        kwargs = {"method": method}
        if method in ("idw", "gaussian"):
            kwargs["k"] = expected_k
        elif method == "local_rbf":
            kwargs["k"] = expected_k
        W = regridder.build_weights(**kwargs)
        assert W.shape == (receiver.num_nodes, sender.num_nodes)
        assert W.nnz == receiver.num_nodes * expected_k

    def test_barycentric_shape(self, regridder, meshes):
        sender, receiver = meshes
        W = regridder.build_weights(method="barycentric")
        assert W.shape == (receiver.num_nodes, sender.num_nodes)
        assert W.nnz == receiver.num_nodes * 3


class TestPartitionOfUnity:
    """Rows should sum to 1 for all methods."""

    @pytest.mark.parametrize("method,kwargs", [
        ("nearest", {}),
        ("idw", {"k": 5}),
        ("gaussian", {"k": 5}),
        ("barycentric", {}),
        ("local_rbf", {"k": 8, "degree": 0}),
        ("local_rbf", {"k": 8, "degree": 1}),
    ])
    def test_row_sums(self, regridder, method, kwargs):
        W = regridder.build_weights(method=method, **kwargs)
        row_sums = np.array(W.sum(axis=1)).ravel()
        assert_allclose(row_sums, 1.0, atol=1e-10)


class TestConstantReproduction:
    """Transferring a constant field should give the same constant."""

    @pytest.mark.parametrize("method,kwargs", [
        ("nearest", {}),
        ("idw", {"k": 5}),
        ("gaussian", {"k": 5}),
        ("barycentric", {}),
        ("local_rbf", {"k": 8, "degree": 0}),
        ("local_rbf", {"k": 8, "degree": 1}),
    ])
    def test_constant(self, regridder, meshes, method, kwargs):
        sender, _ = meshes
        values = np.full(sender.num_nodes, 7.0)
        W = regridder.build_weights(method=method, **kwargs)
        result = W @ values
        assert_allclose(result, 7.0, atol=1e-10)


class TestLinearReproduction:
    """Methods with degree>=1 should reproduce linear fields exactly."""

    @pytest.mark.parametrize("coord", [0, 1, 2])
    def test_local_rbf_linear(self, regridder, meshes, coord):
        sender, receiver = meshes
        values = sender.nodes[:, coord]
        expected = receiver.nodes[:, coord]
        W = regridder.build_weights(method="local_rbf", k=8, degree=1)
        result = W @ values
        assert_allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("coord", [0, 1, 2])
    def test_barycentric_linear(self, regridder, meshes, coord):
        sender, receiver = meshes
        values = sender.nodes[:, coord]
        expected = receiver.nodes[:, coord]
        W = regridder.build_weights(method="barycentric")
        result = W @ values
        assert_allclose(result, expected, atol=1e-2)


class TestRMSERegression:
    """Verify RMSE stays within expected bounds."""

    def test_local_rbf_rmse(self, regridder, meshes):
        sender, receiver = meshes
        tp_s = xyz_to_thetaphi(sender.nodes)
        tp_r = xyz_to_thetaphi(receiver.nodes)
        values = np.cos(3 * tp_s[:, 0]) * np.sin(2 * tp_s[:, 1])
        expected = np.cos(3 * tp_r[:, 0]) * np.sin(2 * tp_r[:, 1])

        W = regridder.build_weights(method="local_rbf", k=8, degree=1)
        result = W @ values
        rmse = np.sqrt(np.mean((result - expected) ** 2))
        assert rmse < 0.2


class TestTransform:
    """Test the transform / @ interface."""

    def test_transform(self, meshes):
        sender, receiver = meshes
        t = MeshTransfer(sender, receiver, method="idw", k=5)
        values = np.ones(sender.num_nodes)
        result = t.transform(values)
        assert_allclose(result, 1.0, atol=1e-10)

    def test_matmul(self, meshes):
        sender, receiver = meshes
        t = MeshTransfer(sender, receiver, method="idw", k=5)
        values = np.ones(sender.num_nodes)
        result = t @ values
        assert_allclose(result, 1.0, atol=1e-10)

    def test_lazy_build(self, meshes):
        sender, receiver = meshes
        t = MeshTransfer(sender, receiver, method="nearest", k=1)
        assert t._weights is None
        _ = t.transform(np.ones(sender.num_nodes))
        assert t._weights is not None

    def test_weights_property(self, meshes):
        sender, receiver = meshes
        t = MeshTransfer(sender, receiver, method="nearest", k=1)
        W = t.weights  # should trigger lazy build
        assert W.shape == (receiver.num_nodes, sender.num_nodes)

    def test_repr(self, meshes):
        sender, receiver = meshes
        t = MeshTransfer(sender, receiver, method="local_rbf", k=8, degree=0)
        r = repr(t)
        assert "MeshTransfer" in r
        assert "not built" in r
        t.build_weights()
        assert "built" in repr(t)


class TestLocalRBFMatchesScipy:
    """Verify local_rbf produces same results as scipy RBFInterpolator."""

    def test_matches_scipy(self, regridder, meshes):
        sender, receiver = meshes
        tp = xyz_to_thetaphi(sender.nodes)
        values = np.cos(3 * tp[:, 0]) * np.sin(2 * tp[:, 1])

        W = regridder.build_weights(
            method="local_rbf", k=8, kernel="thin_plate_spline", degree=1
        )
        sparse_result = W @ values

        scipy_result = RBFInterpolator(
            sender.nodes, values,
            kernel="thin_plate_spline", neighbors=8,
        )(receiver.nodes)

        # Minimal regularization introduces small differences vs
        # scipy's unregularized solve.
        assert_allclose(sparse_result, scipy_result, atol=1e-4)
