import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.prepared import prep
import shapely.geometry as sgeom

from sphedron.mesh import Mesh


def mask_land_vertices(mesh: Mesh):
    vertices_latlon = mesh.vertices_latlon
    land_shp_fname = shpreader.natural_earth(
        resolution="10m", category="physical", name="land"
    )
    land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
    land = prep(land_geom)
    mesh.exclude_vertices(
        np.array([land.contains(sgeom.Point(*latlon)) for latlon in vertices_latlon])
    )
    return mesh


def get_vertex_land_mask(self):
    land_shp_fname = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )
    land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
    land = prep(land_geom)
    return np.array(
        [land.contains(sgeom.Point(*latlon)) for latlon in self.vertices_latlon]
    )
