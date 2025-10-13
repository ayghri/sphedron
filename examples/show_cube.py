from sphedron.cube import Cubesphere
import random
import numpy as np
from trimesh import Scene, load_path, path
import sys

# ic.configureOutput(prefix="hello -> ")
# import random
import matplotlib.pyplot as plt

# mesh = StratifiedIcospheres([1, 2])
# mesh = Icosphere(2)
# mesh = CubeSphere(depth=3, use_length=False)
# mesh = CubeSphere(depth=3, rotate=True)

if __name__ == "__main__":
    depth = int(sys.argv[1])
    normalize = int(sys.argv[2]) > 0
    print(sys.argv[2])
    print(depth, normalize)
    mesh = Cubesphere(refine_factor=depth, rotate=False, refine_by_angle=False, normalize=normalize)

    # mesh.exclude_faces_by_vertex(
    #     np.logical_or(
    #         mesh.vertices[:, 0] < 0, mesh.vertices[:, 1] < 0, mesh.vertices[:, 2] < 0
    #     )
    # )
    # print(mesh.faces)
    # print(mesh.vertices)

    # print(mesh.faces.shape)
    # print(mesh.vertices.shape)
    num_faces = mesh.num_faces
    cmap = plt.get_cmap("gist_ncar")
    face_colors = [
        [int(k * 255) for k in cmap(v)[:3]] + [200]
        for v in np.linspace(0.0, 1.0, num_faces // 2)
    ]
    # face_colors = [
    #     list(a) + [255]
    #     for a in (255.0 * cmap(np.linspace(0.0, 1.0, num_faces))).astype(int)
    # ]
    # print(face_colors)
    # # print(cmap(np.linspace(0.0, 1.0, num_faces)))
    # # print([cmap(v) for v in np.linspace(0.0, 1.0, num_faces)])
    # # print(face_colors)
    random.seed(42)
    random.shuffle(face_colors)
    # # print(face_colors)
    trimesh = mesh.build_trimesh()
    # labels = []
    # for i in range(len(trimesh.vertices)):
    #     p = path.entities.Text(origin=i, text=f"{i}")
    #     labels.append(p)
    # # print(colors)
    #
    # # print(trimesh.visual.face_colors)
    # trimesh.visual.face_colors = [[(250 * i) // num_faces for i in range(1, num_faces + 1)]]
    # face_colors = []
    # for i in range(num_faces):
    # #     face_colors.append(
    # #         [
    # #             # 255 - (i * 255) % 255,
    # #             255 - (i * 255) // num_faces,
    # #             # (i * 255) % 255,
    # #             (i * 255) // num_faces,
    # #             # 255 - (i * 255) % 255,
    # #             255 - (i * 255) // num_faces,
    # #             255,
    # #         ]
    # #     )
    # # edges = trimesh.load_path(mesh.edges_unique.reshape(-1, 2))
    # edges = mesh.edges_unique.T
    # # Create a scene and add both the original mesh and the edge mesh
    #
    # # Show the scene
    # # print("Watertight:", trimesh.is_watertight)
    # # trimesh.fill_holes()
    trimesh.fix_normals()
    # # print("Watertight:", trimesh.is_watertight)
    # face_colors = face_colors[: len(face_colors) // 2]
    face_colors = face_colors + face_colors
    trimesh.visual.face_colors = face_colors
    # # trimesh.show()
    #
    # edge_lines = []
    #
    # # Create line segments from the edges
    # for edge in edges:
    #     line = trimesh.vertices[edge]
    #     edge_lines.append(load_path(line))
    #
    # # Combine all line paths into a single scene
    # # scene = Scene([trimesh] + edge_lines)
    # scene = Scene([trimesh, labels])
    # scene.show()
    trimesh.show(smooth=False, line_settings={"point_size": 0.1, "line_width": 2.0})
