import numpy as np
from sphedron.utils.transform import senders_xyz_in_receivers_rotation


# generate tests for senders_xyz_in_receivers_rotation
def test_senders_xyz_in_receivers_rotation(
    receivers_thetaphi, senders_xyz, zero_latitude, zero_longitude, expected
):
    obtained = senders_xyz_in_receivers_rotation(
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


receivers_thetaphi = np.array([[np.pi / 2, 0], [np.pi / 2, np.pi / 2], [0, np.pi / 2]])
senders_xyz = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
)
# zero_latitude = True
# zero_longitude = True
test_senders_xyz_in_receivers_rotation(
    receivers_thetaphi,
    senders_xyz,
    True,
    True,
    expected=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
)

# expected output
test_senders_xyz_in_receivers_rotation(
    receivers_thetaphi,
    senders_xyz,
    False,
    True,
    expected=np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    ),
)
# expected output

test_senders_xyz_in_receivers_rotation(
    receivers_thetaphi,
    senders_xyz,
    True,
    False,
    expected=np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ),
)

try:
    test_senders_xyz_in_receivers_rotation(
        receivers_thetaphi, senders_xyz, False, False, None
    )
except Exception as e:
    print("Exception captured", e, ". Test passed!")
