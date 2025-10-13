# pylint: disable=C0415
"""
License: Non-Commercial Use Only

Permission is granted to use, copy, modify, and distribute this software
for non-commercial purposes only, with attribution to the original author.
Commercial use requires explicit permission.

This software is provided "as is", without warranty of any kind.
"""

import numpy as np
from sphedron.mesh.base import Mesh


def get_mesh_landmask(mesh: Mesh):
    """Return a land mask for mesh.nodes, where mask[i]==True for land node[i]

    Args:
        mesh: mesh to get the nodes from

    Returns: Array mask of shape (mesh.num_nodes)

    """
    from cartopy.io import shapereader
    from shapely import geometry
    from shapely.ops import unary_union
    from shapely.prepared import prep

    land_shp_fname = shapereader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )
    land_geom = unary_union(
        list(shapereader.Reader(land_shp_fname).geometries())
    )
    land = prep(land_geom)
    mask = [
        land.contains(geometry.Point(*latlong[::-1]))
        for latlong in mesh.nodes_latlong
    ]
    return np.array(mask)


def plot_3d_mesh(
    mesh: Mesh,
    color_faces: bool = False,
    title: str = "Mesh",
    scatter: bool = False,
    s: float = 0.1,
):
    """3d plot of the mesh

    Args:
        mesh:  mesh to display
        color_faces: : whether to color faces
        title: Title of the plot
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.pyplot import get_cmap
    import matplotlib.colors

    nodes, faces = mesh.nodes, mesh.faces
    fig = plt.figure(figsize=(15, 10))
    poly = Poly3DCollection(nodes[faces])

    if color_faces:
        n = len(faces)
        jet = get_cmap("tab20")(np.linspace(0, 1, n))
        jet = np.tile(jet[:, :3], (1, n // n))
        jet = jet.reshape(n, 1, 3)
        face_normals = -np.cross(
            nodes[faces[:, 1]] - nodes[faces[:, 0]],
            nodes[faces[:, 2]] - nodes[faces[:, 0]],
        )
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
        light_source = matplotlib.colors.LightSource(azdeg=59, altdeg=30)
        intensity = light_source.shade_normals(face_normals)

        # blending face colors and face shading intensity
        rgb = np.array(
            light_source.blend_hsv(
                rgb=jet, intensity=intensity.reshape(-1, 1, 1)
            )
        )
        # adding alpha value, may be left out
        rgba = np.concatenate(
            (rgb, 1.0 * np.ones(shape=(rgb.shape[0], 1, 1))), axis=2
        )
        poly.set_facecolor(rgba.reshape(-1, 4))
    # creating mesh with given face colors
    poly.set_edgecolor("black")
    poly.set_linewidth(0.25)

    # and now -- visualization!
    # ax = Axes3D(fig)
    ax = fig.add_subplot(projection="3d")  # type: ignore
    ax.add_collection3d(poly)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_title(title)
    if scatter:
        ax.scatter(
            mesh.nodes[:, 0],
            mesh.nodes[:, 1],
            mesh.nodes[:, 2],
            s=s,
        )
    plt.show()
    plt.close()


def plot_2d_mesh(
    mesh: Mesh,
    title: str = "Mesh",
    scatter: bool = False,
    s: float = 0.1,
):
    """2d Plot of the mesh

    Args:
        mesh: mesh to display
        title: plot title
        scatter: set to True to scatter plot the nodes
        s: scattered nodes size
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import cartopy.crs as ccrs
    from cartopy import feature

    plt.figure(figsize=(10, 16))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    # if edges is None:
    # edges = mesh.edges_unique
    edges = mesh.edges_unique
    segments = mesh.nodes_latlong[:, ::-1][edges]
    lc = LineCollection(
        segments,
        linewidths=0.5,
        transform=ccrs.Geodetic(),
    )
    ax.add_collection(lc)
    ax.add_feature(feature.LAND, facecolor="grey", alpha=0.5, edgecolor="black")
    ax.gridlines(draw_labels=True, linestyle="--", color="black", linewidth=0.5)
    if scatter:
        ax.scatter(
            mesh.nodes_latlong[:, 1],
            mesh.nodes_latlong[:, 0],
            s=s,
            transform=ccrs.PlateCarree(),
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.title(title)
    plt.show()


def plot_nodes(
    mesh: Mesh,
    title: str = "Mesh nodes",
    figsize=(15, 10),
):
    """3d plot of the nodes"""

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)
    poly = Poly3DCollection(mesh.nodes[mesh.faces])

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
    for i, m in enumerate(
        mesh.nodes
    ):  # plot each point + it's index as text above
        ax.scatter(m[0], m[1], m[2], color="b")
        ax.text(m[0], m[1], m[2], str(i), size=20, zorder=10, color="k")
    ax.set_title(title)

    plt.show()
    plt.close()
