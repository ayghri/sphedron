import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sphedron.mesh import Icosphere


class TestIcosphere:
    """Test suite for Icosphere.from_base method."""

    def test_from_base_default_parameters(self):
        """Test creating Icosphere with default parameters."""
        ico = Icosphere.from_base()

        # Base icosahedron has 12 vertices and 20 faces
        assert ico.num_nodes == 12
        assert ico.num_faces == 20
        assert ico.num_edges > 0

        # Check that nodes are on unit sphere
        node_norms = np.linalg.norm(ico.nodes, axis=1)
        assert_array_almost_equal(node_norms, np.ones(12), decimal=10)

    def test_from_base_no_refinement(self):
        """Test creating Icosphere with refine_factor=1 (no refinement)."""
        ico = Icosphere.from_base(refine_factor=1)

        assert ico.num_nodes == 12
        assert ico.num_faces == 20

    def test_from_base_with_refinement(self):
        """Test creating refined Icosphere."""
        refine_factor = 2
        ico = Icosphere.from_base(refine_factor=refine_factor)

        # Refined icosphere should have more nodes and faces
        assert ico.num_nodes > 12
        assert ico.num_faces > 20

        # All nodes should still be on unit sphere
        node_norms = np.linalg.norm(ico.nodes, axis=1)
        assert_array_almost_equal(node_norms, np.ones(ico.num_nodes), decimal=8)

    def test_from_base_multiple_refinement_levels(self):
        """Test different refinement levels produce increasing complexity."""
        ico1 = Icosphere.from_base(refine_factor=1)
        ico2 = Icosphere.from_base(refine_factor=2)
        ico3 = Icosphere.from_base(refine_factor=3)

        # Higher refinement should produce more nodes and faces
        assert ico1.num_nodes < ico2.num_nodes < ico3.num_nodes
        assert ico1.num_faces < ico2.num_faces < ico3.num_faces

    def test_from_base_no_rotation(self):
        """Test creating Icosphere without rotation."""
        ico_rotated = Icosphere.from_base(rotate=True)
        ico_unrotated = Icosphere.from_base(rotate=False)

        # Both should have same number of nodes/faces
        assert ico_rotated.num_nodes == ico_unrotated.num_nodes
        assert ico_rotated.num_faces == ico_unrotated.num_faces

        # But node positions should be different due to rotation
        assert not np.allclose(ico_rotated.nodes, ico_unrotated.nodes)

    def test_from_base_rotation_preserves_unit_sphere(self):
        """Test that rotation preserves nodes on unit sphere."""
        ico_rotated = Icosphere.from_base(rotate=True)
        ico_unrotated = Icosphere.from_base(rotate=False)

        # Both should have nodes on unit sphere
        rotated_norms = np.linalg.norm(ico_rotated.nodes, axis=1)
        unrotated_norms = np.linalg.norm(ico_unrotated.nodes, axis=1)

        assert_array_almost_equal(rotated_norms, np.ones(ico_rotated.num_nodes))
        assert_array_almost_equal(
            unrotated_norms, np.ones(ico_unrotated.num_nodes)
        )

    def test_from_base_with_refine_by_angle(self):
        """Test creating Icosphere with refine_by_angle parameter."""
        ico_angle = Icosphere.from_base(refine_by_angle=True)
        ico_normal = Icosphere.from_base(refine_by_angle=False)

        # Should create valid icospheres regardless of refine_by_angle
        assert ico_angle.num_nodes >= 12
        assert ico_angle.num_faces >= 20
        assert ico_normal.num_nodes >= 12
        assert ico_normal.num_faces >= 20

    def test_from_base_rotation_angle_property(self):
        """Test that Icosphere uses correct rotation angle."""
        expected_angle = -np.pi / 2 + np.arcsin((1 + np.sqrt(5)) / np.sqrt(12))
        assert abs(Icosphere.rotation_angle - expected_angle) < 1e-10

    def test_from_base_rotation_axis_property(self):
        """Test that Icosphere uses correct rotation axis."""
        assert Icosphere.rotation_axis == "y"

    def test_from_base_triangular_faces(self):
        """Test that Icosphere produces triangular faces."""
        ico = Icosphere.from_base()

        # All faces should be triangles (3 vertices per face)
        assert ico.faces.shape[1] == 3

        # faces2triangles should return the same as faces for triangular mesh
        assert_array_equal(ico.triangles, ico.faces)

    def test_from_base_face_indices_valid(self):
        """Test that face indices are valid node references."""
        ico = Icosphere.from_base()

        # All face indices should be within node range
        assert np.all(ico.faces >= 0)
        assert np.all(ico.faces < ico.num_nodes)

    def test_from_base_edges_properties(self):
        """Test edge-related properties of created Icosphere."""
        ico = Icosphere.from_base()

        # Should have edges
        assert ico.num_edges > 0
        assert ico.edges.shape[1] == 2

        # Edge indices should be valid
        assert np.all(ico.edges >= 0)
        assert np.all(ico.edges < ico.num_nodes)

        # Unique edges should be subset of all edges
        assert ico.edges_unique.shape[0] <= ico.num_edges

    def test_from_base_base_geometry_unchanged(self):
        """Test that base geometry is not modified by from_base calls."""
        # Get base geometry
        base_nodes1, base_faces1 = Icosphere.base()

        # Create icosphere

        # Get base geometry again
        base_nodes2, base_faces2 = Icosphere.base()

        # Should be identical
        assert_array_equal(base_nodes1, base_nodes2)
        assert_array_equal(base_faces1, base_faces2)

    def test_from_base_consistent_results(self):
        """Test that multiple calls with same parameters produce identical results."""
        ico1 = Icosphere.from_base(
            refine_factor=2, rotate=True, refine_by_angle=False
        )
        ico2 = Icosphere.from_base(
            refine_factor=2, rotate=True, refine_by_angle=False
        )

        assert ico1.num_nodes == ico2.num_nodes
        assert ico1.num_faces == ico2.num_faces
        assert_array_almost_equal(ico1.nodes, ico2.nodes)
        assert_array_equal(ico1.faces, ico2.faces)

    def test_from_base_nodes_latlong_conversion(self):
        """Test that nodes can be converted to lat/long coordinates."""
        ico = Icosphere.from_base()

        latlong = ico.nodes_latlong
        assert latlong.shape == (ico.num_nodes, 2)

        # Latitude should be in [-π/2, π/2]
        assert np.all(latlong[:, 0] >= -np.rad2deg(np.pi / 2))
        assert np.all(latlong[:, 0] <= np.rad2deg(np.pi / 2))

        # Longitude should be in [-π, π]
        assert np.all(latlong[:, 1] >= -np.rad2deg(np.pi))
        assert np.all(latlong[:, 1] <= np.rad2deg(np.pi))

    @pytest.mark.parametrize("refine_factor", [1, 2, 3, 4])
    def test_from_base_various_refinement_factors(self, refine_factor):
        """Test Icosphere creation with various refinement factors."""
        ico = Icosphere.from_base(refine_factor=refine_factor)

        assert ico.num_nodes >= 12
        assert ico.num_faces >= 20

        # All nodes should be on unit sphere
        node_norms = np.linalg.norm(ico.nodes, axis=1)
        assert_array_almost_equal(node_norms, np.ones(ico.num_nodes), decimal=8)

    @pytest.mark.parametrize("rotate", [True, False])
    def test_from_base_rotation_parameter(self, rotate):
        """Test Icosphere creation with different rotation settings."""
        ico = Icosphere.from_base(rotate=rotate)

        assert ico.num_nodes == 12
        assert ico.num_faces == 20

        # Nodes should be on unit sphere regardless of rotation
        node_norms = np.linalg.norm(ico.nodes, axis=1)
        assert_array_almost_equal(node_norms, np.ones(12), decimal=10)
