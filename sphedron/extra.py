import numpy as np
from sphedron.mesh import Mesh


def mask_land_nodes(mesh: Mesh):
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union
    from shapely.prepared import prep
    import shapely.geometry as sgeom

    nodes_latlon = mesh.nodes_latlon
    land_shp_fname = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )
    land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
    land = prep(land_geom)
    mask = [land.contains(sgeom.Point(*latlon[::-1])) for latlon in nodes_latlon]
    mesh.mask_nodes(np.array(mask))
    return mesh


def get_land_mask(nodes_latlon):

    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union
    from shapely.prepared import prep
    import shapely.geometry as sgeom

    land_shp_fname = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )
    land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
    land = prep(land_geom)
    mask = [land.contains(sgeom.Point(*latlon[::-1])) for latlon in nodes_latlon]
    return np.array(mask)


def get_vertex_land_mask(self):
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union
    from shapely.prepared import prep
    import shapely.geometry as sgeom

    land_shp_fname = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )
    land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
    land = prep(land_geom)
    return np.array(
        [land.contains(sgeom.Point(*latlon)) for latlon in self.nodes_latlon]
    )


def plot_3d_mesh(
    mesh: Mesh,
    # nodes,
    # faces,
    color_faces=False,
    title="Icosphere",
):
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from matplotlib.pyplot import get_cmap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def plot_2d_mesh(
    mesh: Mesh, edges=None, title="Icosphere", resolution=110, scatter=False, s=0.1
):

    import matplotlib.pyplot as plt
    from cartopy import feature
    from matplotlib.collections import LineCollection
    import cartopy.crs as ccrs

    plt.figure(figsize=(10, 16))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    if edges is None:
        edges = mesh.edges_unique
    segments = mesh.nodes_latlon[:, ::-1][edges]
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
            mesh.nodes_latlon[:, 1],
            mesh.nodes_latlon[:, 0],
            s=s,
            transform=ccrs.PlateCarree(),
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.title(title)
    plt.show()


def plot_nodes(
    mesh: Mesh,
    title="Mesh nodes",
    figsize=(15, 10),
):

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

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
    for i, m in enumerate(mesh.nodes):  # plot each point + it's index as text above
        ax.scatter(m[0], m[1], m[2], color="b")
        ax.text(m[0], m[1], m[2], str(i), size=20, zorder=10, color="k")
    ax.set_title(title)

    plt.show()
    plt.close()
