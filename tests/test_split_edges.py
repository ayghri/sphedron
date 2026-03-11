import pytest
import numpy as np
from numpy.testing import assert_allclose
from sphedron.refine import split_edges, rectangle_interior
from sphedron import UniformMesh
from sphedron.mesh.base import RectangularMesh
import sphedron.transform as _transform


class TestSplitEdges:
    """Tests for split_edges with use_angle=True vs use_angle=False."""

    @pytest.fixture
    def unit_edge(self):
        """Two points on the unit sphere separated by a known angle."""
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])  # 90 degrees from A
        return np.array([[A, B]])

    def test_linear_split_unequal_arc_lengths(self, unit_edge):
        """use_angle=False: linear interpolation produces unequal arc lengths."""
        pts = split_edges(unit_edge, num_segments=5, use_angle=False)
        # Normalize to project onto sphere
        pts_norm = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        all_pts = np.vstack([unit_edge[0, 0], pts_norm, unit_edge[0, 1]])
        # Compute arc angles between consecutive points
        dots = np.sum(all_pts[:-1] * all_pts[1:], axis=1)
        angles = np.arccos(np.clip(dots, -1, 1))
        # Linear interpolation does NOT produce equal arc lengths
        assert not np.allclose(angles, angles[0], atol=1e-6)

    def test_angle_split_equal_arc_lengths(self, unit_edge):
        """use_angle=True: geodesic interpolation produces equal arc lengths."""
        pts = split_edges(unit_edge, num_segments=5, use_angle=True)
        pts_norm = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        all_pts = np.vstack([unit_edge[0, 0], pts_norm, unit_edge[0, 1]])
        # Compute arc angles between consecutive points
        dots = np.sum(all_pts[:-1] * all_pts[1:], axis=1)
        angles = np.arccos(np.clip(dots, -1, 1))
        # Geodesic interpolation produces equal arc lengths
        assert_allclose(angles, angles[0], atol=1e-12)

    def test_angle_split_expected_angle(self, unit_edge):
        """use_angle=True: each segment subtends the correct angle."""
        n_segments = 4
        pts = split_edges(unit_edge, num_segments=n_segments, use_angle=True)
        pts_norm = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        all_pts = np.vstack([unit_edge[0, 0], pts_norm, unit_edge[0, 1]])
        dots = np.sum(all_pts[:-1] * all_pts[1:], axis=1)
        angles = np.arccos(np.clip(dots, -1, 1))
        expected_angle = np.pi / 2 / n_segments  # 90 degrees / n_segments
        assert_allclose(angles, expected_angle, atol=1e-12)

    def test_linear_split_equal_euclidean_spacing(self, unit_edge):
        """use_angle=False: points are equally spaced in Euclidean distance."""
        pts = split_edges(unit_edge, num_segments=5, use_angle=False)
        # Points are NOT normalized -- check raw Euclidean spacing
        all_pts = np.vstack([unit_edge[0, 0], pts, unit_edge[0, 1]])
        dists = np.linalg.norm(np.diff(all_pts, axis=0), axis=1)
        assert_allclose(dists, dists[0], atol=1e-12)

    def test_both_modes_produce_same_count(self, unit_edge):
        """Both modes return the same number of points."""
        n = 6
        pts_linear = split_edges(unit_edge, num_segments=n, use_angle=False)
        pts_angle = split_edges(unit_edge, num_segments=n, use_angle=True)
        assert pts_linear.shape == pts_angle.shape == (n - 1, 3)

    def test_results_differ_between_modes(self, unit_edge):
        """The two modes produce different coordinates."""
        pts_linear = split_edges(unit_edge, num_segments=5, use_angle=False)
        pts_angle = split_edges(unit_edge, num_segments=5, use_angle=True)
        assert not np.allclose(pts_linear, pts_angle)

    @pytest.mark.parametrize("n_segments", [2, 3, 5, 10])
    def test_angle_split_various_segments(self, unit_edge, n_segments):
        """use_angle=True produces equal arcs for various segment counts."""
        pts = split_edges(unit_edge, num_segments=n_segments, use_angle=True)
        pts_norm = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        all_pts = np.vstack([unit_edge[0, 0], pts_norm, unit_edge[0, 1]])
        dots = np.sum(all_pts[:-1] * all_pts[1:], axis=1)
        angles = np.arccos(np.clip(dots, -1, 1))
        assert_allclose(angles, angles[0], atol=1e-12)


class TestRectangleInterior:
    """Tests for rectangle_interior use_angle parameter."""

    def test_default_is_linear(self):
        """Default use_angle=False uses linear interpolation."""
        ad = np.array([[0.8, 0.6, 0.0], [0.6, 0.8, 0.0]])
        bc = np.array([[0.0, 0.6, 0.8], [0.0, 0.8, 0.6]])
        result_default = rectangle_interior(ad, bc)
        result_linear = rectangle_interior(ad, bc, use_angle=False)
        assert_allclose(result_default, result_linear)

    def test_angle_differs_from_linear(self):
        """use_angle=True produces different results than use_angle=False."""
        ad = np.array([[0.8, 0.6, 0.0], [0.6, 0.8, 0.0]])
        bc = np.array([[0.0, 0.6, 0.8], [0.0, 0.8, 0.6]])
        result_linear = rectangle_interior(ad, bc, use_angle=False)
        result_angle = rectangle_interior(ad, bc, use_angle=True)
        assert not np.allclose(result_linear, result_angle)


class TestUniformMeshRefinement:
    """Test that refining a UniformMesh by factor 2 reproduces a finer grid."""

    def test_refined_shape_matches_fine_grid(self):
        """Refining 1deg mesh by 2 gives same node/face count as aligned 0.5deg."""
        coarse = UniformMesh(resolution=1.0)
        fine_lats = np.arange(-89.5, 90, 0.5)
        fine_longs = np.arange(0.5, 360, 0.5)
        fine = UniformMesh(uniform_lats=fine_lats, uniform_longs=fine_longs)

        refined_nodes, refined_faces = RectangularMesh.refine(
            coarse._all_nodes, coarse._all_faces, factor=2, use_angle=True
        )

        assert refined_nodes.shape[0] == fine._all_nodes.shape[0]
        assert refined_faces.shape[0] == fine._all_faces.shape[0]

    def test_meridian_edges_exact(self):
        """Slerp refinement is exact along meridians (great circles).

        Meridian edges connect nodes at adjacent latitudes on the same
        longitude.  Since meridians are great circles, slerp midpoints
        should land at exactly the intermediate latitude.
        """
        coarse = UniformMesh(resolution=1.0)
        refined_nodes, _ = RectangularMesh.refine(
            coarse._all_nodes, coarse._all_faces, factor=2, use_angle=True
        )

        # Build the expected fine grid
        fine_lats = np.arange(-89.5, 90, 0.5)
        fine_longs = np.arange(0.5, 360, 0.5)
        fine = UniformMesh(uniform_lats=fine_lats, uniform_longs=fine_longs)

        # Convert both to lat/lon
        refined_ll = _transform.xyz_to_latlong(refined_nodes)
        fine_ll = _transform.xyz_to_latlong(fine._all_nodes)

        # For each coarse longitude, extract refined nodes on that meridian
        for lon in coarse.uniform_longs[:5]:  # spot-check first 5 meridians
            # Refined nodes on this meridian (lon matches to high precision)
            ref_mask = np.abs(refined_ll[:, 1] - lon) < 1e-8
            fine_mask = np.abs(fine_ll[:, 1] - lon) < 1e-8

            ref_lats = np.sort(refined_ll[ref_mask, 0])
            fine_lats_on_meridian = np.sort(fine_ll[fine_mask, 0])

            assert ref_lats.shape == fine_lats_on_meridian.shape, (
                f"Meridian lon={lon}: count mismatch "
                f"{ref_lats.shape} vs {fine_lats_on_meridian.shape}"
            )
            assert_allclose(
                ref_lats, fine_lats_on_meridian, atol=1e-10,
                err_msg=f"Meridian lon={lon}: latitudes differ",
            )
