# from sphedron import Icospherr
import sphedron as sph
import random
import numpy as np
from trimesh import Scene, load_path, creation
import sys

# import random
import matplotlib.pyplot as plt

# mesh = StratifiedIcospheres([1, 2])
if __name__ == "__main__":
    depth = int(sys.argv[1])
    rotate = int(sys.argv[2]) > 0
    print(sys.argv)
    # mesh = Icosphere(depth, rotate=rotate, normalize=normalize)
    mesh1 = sph.Icosphere(depth, rotate=rotate)

    def show_mesh(mesh):
        print(mesh)
        num_faces = mesh.num_faces
        cmap = plt.get_cmap("gist_ncar")
        face_colors = [
            [int(k * 255) for k in cmap(v)[:3]] + [255]
            for v in np.linspace(0.0, 1.0, num_faces)
        ]
        random.seed(42)
        random.shuffle(face_colors)

        axis = creation.axis(
            origin_size=0.05, axis_length=1, axis_radius=0.01
        )  # Customize if needed
        trimesh = mesh.build_trimesh()
        edges = mesh.edges_unique
        trimesh.fix_normals()
        trimesh.visual.face_colors = face_colors

        edge_lines = []

        # Create line segments from the edges
        # for edge in edges:
            # line = trimesh.vertices[edge]
            # edge_lines.append(load_path(line))

        # Combine all line paths into a single scene
        # scene = Scene([trimesh] + edge_lines)
        scene = Scene([trimesh, axis])
        scene.show(smooth=False, line_settings={"point_size": 0.1, "line_width": 2.0})
        # trimesh.show(smooth=False, line_settings={"point_size": 0.1, "line_width": 2.0})

    from sphedron.utils.geo import get_land_mask

    show_mesh(mesh1)
    mesh2 = sph.NestedIcospheres(factors=[1, 4, 4, 4])
    show_mesh(mesh2)
    land_mask = get_land_mask(nodes_latlong=mesh2.nodes_latlong)
    mesh2.mask_nodes(land_mask)
    show_mesh(mesh2)

# Create a scene with both the mesh and axis
# scene = trimesh.Scene([mesh, axis])
