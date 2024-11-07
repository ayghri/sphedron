from matplotlib.collections import LineCollection
from matplotlib.pyplot import get_cmap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy import feature
from sphedron.mesh import Mesh


def plot_3d_mesh(
    # mesh: Mesh,
    vertices,
    faces,
    color_faces=False,
    title="Icosphere",
):
    # vertices, faces = mesh.vertices, mesh.faces
    fig = plt.figure(figsize=(15, 10))
    poly = Poly3DCollection(vertices[faces])

    if color_faces:
        n = len(faces)
        jet = get_cmap("tab20")(np.linspace(0, 1, n))
        jet = np.tile(jet[:, :3], (1, n // n))
        jet = jet.reshape(n, 1, 3)
        face_normals = -np.cross(
            vertices[faces[:n, 1]] - vertices[faces[:n, 0]],
            vertices[faces[:n, 2]] - vertices[faces[:n, 0]],
        )
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
        light_source = matplotlib.colors.LightSource(azdeg=59, altdeg=30)
        intensity = light_source.shade_normals(face_normals)

        # blending face colors and face shading intensity
        rgb = np.array(
            light_source.blend_hsv(rgb=jet, intensity=intensity.reshape(-1, 1, 1))
        )
        # adding alpha value, may be left out
        rgba = np.concatenate((rgb, 1.0 * np.ones(shape=(rgb.shape[0], 1, 1))), axis=2)
        poly.set_facecolor(rgba.reshape(-1, 4))
    # creating mesh with given face colors
    poly.set_edgecolor("black")
    poly.set_linewidth(0.25)

    # and now -- visualization!
    # ax = Axes3D(fig)
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(poly)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_title(title)
    plt.show()
    plt.close()


def plot_2d_mesh(vertices_latlon, edges, title="Icosphere", resolution=110):
    plt.figure(figsize=(10, 16))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    edges = edges.T
    segments = vertices_latlon[edges]
    lc = LineCollection(
        segments,
        linewidths=0.5,
        transform=ccrs.Geodetic(),
    )
    ax.add_collection(lc)
    # ax.add_feature(feature.LAND, facecolor="white")
    # ax.coastlines(resolution=f"{resolution}m", linewidth=1)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # lc = LineCollection(segments, linewidths=0.5, transform=ccrs.Geodetic())
    # ax.add_collection(lc)
    ax.add_feature(feature.LAND, facecolor="grey", alpha=0.5, edgecolor="black")
    ax.gridlines(draw_labels=True, linestyle="--", color="black", linewidth=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.title(title)
    plt.show()


def plot_vertices(
    mesh: Mesh,
    title="Mesh vertices",
    figsize=(15, 10),
):

    fig = plt.figure(figsize=figsize)
    poly = Poly3DCollection(mesh.vertices[mesh.squares])

    poly.set_edgecolor("black")
    poly.set_linewidth(0.25)

    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(poly)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    for i, m in enumerate(mesh.vertices):  # plot each point + it's index as text above
        ax.scatter(m[0], m[1], m[2], color="b")
        ax.text(m[0], m[1], m[2], str(i), size=20, zorder=10, color="k")
    ax.set_title(title)

    plt.show()
    plt.close()
