from sphedron import Icosphere, NestedIcospheres
import numpy as np
from numpy.testing import assert_array_less
from numpy.testing import assert_array_equal

import unittest


class TestMeshQuery(unittest.TestCase):
    description = "test mesh queries based on radius/#neighbors"

    def setUp(self) -> None:
        self.receiver_mesh = Icosphere.from_base(refine_factor=64)
        self.sender_mesh = NestedIcospheres(factors=[1, 2, 2, 2, 2, 2, 2])

    def test_radius(self):
        """Check if query_radius edges are actually within radius"""

        radius = 0.022
        nearest_s2r = self.sender_mesh.query_edges_from_radius(
            receiver_mesh=self.receiver_mesh, radius=radius
        )

        dists = np.linalg.norm(
            self.sender_mesh.nodes[nearest_s2r[:, 0]]
            - self.receiver_mesh.nodes[nearest_s2r[:, 1]],
            axis=1,
        )

        assert_array_less(dists, radius, strict=False)

    def test_neighbors(self):
        """Check if query neighbors for some random receiver node
        edges are among nearest neighbors"""

        n_neighbors = 10
        nearest_s2r = self.sender_mesh.query_edges_from_neighbors(
            receiver_mesh=self.receiver_mesh, n_neighbors=n_neighbors
        )
        s_nodes = np.sort(nearest_s2r[nearest_s2r[:, 1] == 0][:, 0])
        r_node = self.receiver_mesh.nodes[0]
        dists = np.linalg.norm(self.sender_mesh.nodes - r_node, axis=-1)
        nearest_senders = np.sort(np.argsort(dists)[:n_neighbors])
        assert_array_equal(s_nodes, nearest_senders)


tests = [TestMeshQuery]
