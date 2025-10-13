from sphedron import Icosphere
import numpy as np

depth = 32
mesh = Icosphere(refine_factor=depth)


n_neighbors = 5
lons = np.arange(-180, 180)
lats = np.arange(-90, 91)
lons_grid, lats_grid = np.meshgrid(lats, lons)
uniform_grid = np.stack([lons_grid, lats_grid], axis=-1)
print(f"The uniform latlong grid has {uniform_grid.shape} vertices")


mesh_nearest_vertices, mesh_nearest_distances = mesh.nearest_vertices_to_mesh_vertices(
    uniform_grid.reshape(-1, 2), n_neighbors=n_neighbors
)
print(
    f"Indices of the nearest {n_neighbors} vertices from the Icosphere:",
    mesh_nearest_vertices.reshape(*uniform_grid.shape[:-1], 5).shape,
    f"\ndistances of the nearest {n_neighbors} vertices from the Icosphere:",
    mesh_nearest_distances.reshape(*uniform_grid.shape[:-1], 5).shape,
)
