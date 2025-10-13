# from sphedron.helpers import plot_3d_mesh
# from sphedron.helpers import mask_land_vertices

from sphedron import Icosphere
from sphedron.extra import plot_3d_mesh
from sphedron.extra import plot_nodes


depth = 16
mesh = Icosphere(refine_factor=depth)
plot_3d_mesh(mesh, color_faces=True, scatter=True)
# plot_nodes(mesh)
# land_vertices = mask_land_vertices(mesh.vertices_latlong)
# mesh.exclude_faces_by_vertex(land_vertices)
# mesh.exclude_faces_by_vertex(mesh.get_vertex_land_mask())
# print(mesh.vertices_latlong.shape)
# print(mesh.num_faces)
# plot_3d_mesh(mesh.vertices, mesh.faces)
cesm2_path = "/buckets/datasets/ssh/simulations/cesm2/"

grid_data = read_netcdf(Path(cesm2_path).joinpath("grid_info.nc"))

tlongs = grid_data["TLONG"].values
tlats = grid_data["TLAT"].values
cesm2_grid_coords = np.stack([tlats, tlongs], axis=-1)
cesm2_grid_coords = cesm2_grid_coords.reshape(-1, 2)

from sphedron.mesh import NodesOnlyMesh

grid = NodesOnlyMesh(cesm2_grid_coords)
print(grid)
