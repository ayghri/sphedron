from sphedron import Icosphere, NestedIcospheres
from scipy.spatial import cKDTree  # pyright: ignore
from sklearn.neighbors import NearestNeighbors
import time

mesh = NestedIcospheres([1, 2, 2, 2, 2, 2, 2, 2])
print(mesh)

nodes = mesh.nodes
start = time.time()
for _ in range(10):
    # current_indices = cKDTree(nodes).query(x=nodes, k=10, workers=-1)
    current_indices = cKDTree(nodes).query_ball_point(x=nodes, r=0.02, workers=-1)
print(time.time() - start)

time.sleep(5)

start = time.time()
for _ in range(10):
    current_indices = (
        NearestNeighbors(algorithm="kd_tree", n_jobs=-1)
        .fit(nodes)
        .radius_neighbors(nodes, radius=0.02, return_distance=False)
        # .kneighbors(nodes, 10, return_distance=False)
    )
print(time.time() - start)

time.sleep(5)

start = time.time()
for _ in range(10):
    current_indices = cKDTree(nodes).query(x=nodes, k=10, workers=-1)
print(time.time() - start)

time.sleep(5)

start = time.time()
for _ in range(10):
    current_indices = (
        NearestNeighbors(algorithm="kd_tree", n_jobs=-1)
        .fit(nodes)
        .kneighbors(nodes, 10, return_distance=False)
    )
print(time.time() - start)

time.sleep(5)
