import pytest
import numpy as np
from sphedron.transform import rotate_senders_by_receivers
from sphedron.transform import thetaphi_to_xyz
from numpy.testing import assert_array_almost_equal


class TestRotations:
    """Test rotations of sender/receiver based on zero lat/long."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Set up test data for all rotation tests."""
        receivers_thetaphi = np.array(
            [[np.pi / 2, 0], [np.pi / 2, np.pi / 2], [0, 0]]
        )
        self.receivers_xyz = thetaphi_to_xyz(receivers_thetaphi)
        self.senders_xyz = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

    def compare_result(
        self, expected, zero_latitude=False, zero_longitude=False
    ):
        """Helper method to compare rotation results."""
        obtained = rotate_senders_by_receivers(
            self.receivers_xyz,
            self.senders_xyz,
            zero_latitude,
            zero_longitude,
        )
        assert_array_almost_equal(obtained, expected)

    def test_zerolatlong(self):
        """Test rotation with both zero latitude and longitude."""
        expected = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        self.compare_result(expected, True, True)

    def test_zerolat(self):
        """Test rotation with zero latitude only."""
        expected = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        self.compare_result(expected, zero_latitude=True)

    def test_zerolon(self):
        """Test rotation with zero longitude only."""
        expected = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        self.compare_result(expected, zero_longitude=True)

    def test_receiverlatlong(self):
        """Test receiver rotation with both zero latitude and longitude."""
        receivers = np.random.randn(400, 3)
        receivers = receivers / np.linalg.norm(receivers, axis=1, keepdims=True)
        rotated = rotate_senders_by_receivers(receivers, receivers, True, True)
        assert_array_almost_equal(rotated[:, [1, 2]], 0.0)

    def test_receiverlat(self):
        """Test receiver rotation with zero latitude only."""
        receivers = np.random.randn(400, 3)
        receivers = receivers / np.linalg.norm(receivers, axis=1, keepdims=True)
        rotated = rotate_senders_by_receivers(
            receivers, receivers, zero_latitude=True
        )
        assert_array_almost_equal(rotated[:, [2]], 0.0)

    def test_receiverlon(self):
        """Test receiver rotation with zero longitude only."""
        receivers = np.random.randn(400, 3)
        receivers = receivers / np.linalg.norm(receivers, axis=1, keepdims=True)
        rotated = rotate_senders_by_receivers(
            receivers, receivers, zero_longitude=True
        )
        assert_array_almost_equal(rotated[:, [1]], 0)

    @pytest.mark.parametrize("zero_latitude,zero_longitude", [
        (True, False),
        (False, True),
        (True, True),
    ])
    def test_rotation_parameters(self, zero_latitude, zero_longitude):
        """Test rotation with different parameter combinations."""
        result = rotate_senders_by_receivers(
            self.receivers_xyz,
            self.senders_xyz,
            zero_latitude,
            zero_longitude,
        )
        # Basic sanity check - result should have same shape
        assert result.shape == self.senders_xyz.shape
        # Result should be normalized (on unit sphere)
        norms = np.linalg.norm(result, axis=1)
        assert_array_almost_equal(norms, 1.0)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError):
            # Should raise error when both zero_latitude and zero_longitude are False
            rotate_senders_by_receivers(
                self.receivers_xyz,
                self.senders_xyz,
                zero_latitude=False,
                zero_longitude=False,
            )

    def test_single_point_rotation(self):
        """Test rotation with single receiver and sender points."""
        single_receiver = self.receivers_xyz[:1]
        single_sender = self.senders_xyz[:1]
        
        result = rotate_senders_by_receivers(
            single_receiver, single_sender, True, True
        )
        
        assert result.shape == (1, 3)
        # Check normalization
        assert_array_almost_equal(np.linalg.norm(result), 1.0)

    def test_empty_input(self):
        """Test behavior with empty input arrays."""
        empty_array = np.empty((0, 3))
        
        result = rotate_senders_by_receivers(
            empty_array, empty_array, True, True
        )
        
        assert result.shape == (0, 3)
