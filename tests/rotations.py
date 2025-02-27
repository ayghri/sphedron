import numpy as np
from sphedron.transform import rotate_senders_by_receivers
from sphedron.utils import thetaphi_to_xyz
from sphedron.utils import xyz_to_thetaphi


# generate tests for rotate_senders_by_receivers
def test_rotate_senders_by_receivers(
    receivers_thetaphi, senders_xyz, zero_latitude, zero_longitude, expected
):
    obtained = rotate_senders_by_receivers(
        receivers_thetaphi, senders_xyz, zero_latitude, zero_longitude
    )
    if not np.allclose(obtained, expected):
        print("Test failed")
        print("obtained")
        print(obtained)
        print("expected")
        print(expected)
    else:
        print(f"lat/lon={zero_latitude}/{zero_longitude}, Test passed")


receivers_thetaphi = np.array([[np.pi / 2, 0], [np.pi / 2, np.pi / 2], [0, 0]])
receivers_xyz = thetaphi_to_xyz(receivers_thetaphi)
# print(receivers_xyz)
recovered = xyz_to_thetaphi(receivers_xyz)
# print(recovered, receivers_thetaphi)
assert np.allclose(receivers_thetaphi, recovered)
senders_xyz = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
)
# zero_latitude = True
# zero_longitude = True
test_rotate_senders_by_receivers(
    receivers_xyz,
    senders_xyz,
    True,
    True,
    expected=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
)

# expected output
test_rotate_senders_by_receivers(
    receivers_xyz,
    senders_xyz,
    False,
    True,
    expected=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ),
)
# expected output

test_rotate_senders_by_receivers(
    receivers_xyz,
    senders_xyz,
    True,
    False,
    expected=np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
)

try:
    test_rotate_senders_by_receivers(receivers_xyz, senders_xyz, False, False, None)
except Exception as e:
    print("Exception captured", e, ". Test passed!")


import numpy as np


def test_receivers_rotation():

    print("sanity checks")
    receivers = np.random.randn(400, 3)
    receivers = receivers / np.linalg.norm(receivers, axis=1, keepdims=True)
    rotated = rotate_senders_by_receivers(receivers, receivers, True, True)
    print("Checking receivers in receivers lat/lon zeroed", True, True, end=". ")
    # (1, 0, 0)
    assert np.allclose(rotated[:, [1, 2]], 0)
    print("Passed!")
    rotated = rotate_senders_by_receivers(receivers, receivers, zero_latitude=True)
    print("Checking receivers in receivers lat/lon zeroed", True, False, end=". ")
    # (?, ?, 0)
    assert np.allclose(rotated[:, [2]], 0)
    print("Passed!")
    rotated = rotate_senders_by_receivers(receivers, receivers, zero_longitude=True)
    print("Checking receivers in receivers lat/lon zeroed", False, True, end=". ")
    # (?, 0, ?)
    assert np.allclose(rotated[:, [1]], 0)
    print("Passed!")


test_receivers_rotation()
