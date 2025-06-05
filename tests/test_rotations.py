import numpy as np
from sphedron.utils.transform import rotate_senders_by_receivers
from sphedron.utils.transform import thetaphi_to_xyz
from numpy.testing import assert_array_almost_equal

import unittest


class TestRotations(unittest.TestCase):
    """test rotations of sender/receiver based on zero lat/long."""

    def setUp(self):
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
        obtained = rotate_senders_by_receivers(
            self.receivers_xyz,
            self.senders_xyz,
            zero_latitude,
            zero_longitude,
        )
        assert_array_almost_equal(obtained, expected)

    def test_zerolatlong(self):

        expected = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        self.compare_result(expected, True, True)

    def test_zerolat(self):
        expected = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        self.compare_result(expected, zero_latitude=True)

    def test_zerolon(self):
        expected = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        self.compare_result(expected, zero_longitude=True)

    def test_receiverlatlong(self):
        receivers = np.random.randn(400, 3)
        receivers = receivers / np.linalg.norm(receivers, axis=1, keepdims=True)
        rotated = rotate_senders_by_receivers(receivers, receivers, True, True)
        assert_array_almost_equal(rotated[:, [1, 2]], 0.0)

    def test_receiverlat(self):
        receivers = np.random.randn(400, 3)
        receivers = receivers / np.linalg.norm(receivers, axis=1, keepdims=True)
        rotated = rotate_senders_by_receivers(
            receivers, receivers, zero_latitude=True
        )
        assert_array_almost_equal(rotated[:, [2]], 0.0)

    def test_receiverlon(self):
        receivers = np.random.randn(400, 3)
        receivers = receivers / np.linalg.norm(receivers, axis=1, keepdims=True)
        rotated = rotate_senders_by_receivers(
            receivers, receivers, zero_longitude=True
        )
        assert_array_almost_equal(rotated[:, [1]], 0)


tests = [TestRotations]
